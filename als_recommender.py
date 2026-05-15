from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split


@dataclass
class ALSConfig:
    n_factors: int = 32
    n_iters: int = 15
    reg: float = 0.1
    random_state: int = 42


class ExplicitALSRecommender:
    def __init__(self, config: ALSConfig | None = None):
        self.config = config or ALSConfig()
        self.global_mean: float = 0.0
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.user_ids: np.ndarray | None = None
        self.item_ids: np.ndarray | None = None
        self.user_to_idx: dict[int, int] = {}
        self.item_to_idx: dict[int, int] = {}
        self.item_popularity: np.ndarray | None = None
        self._user_item_matrix: csr_matrix | None = None
        self._ratings_by_user: dict[int, set[int]] = {}

    def fit(self, ratings: pd.DataFrame) -> "ExplicitALSRecommender":
        required_columns = {"user_id", "anime_id", "rating"}
        missing_columns = required_columns - set(ratings.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

        data = ratings[["user_id", "anime_id", "rating"]].copy()
        data["rating"] = pd.to_numeric(data["rating"], errors="coerce")
        data = data.dropna(subset=["user_id", "anime_id", "rating"])
        data = data[data["rating"] > 0].copy()

        if data.empty:
            raise ValueError("No positive ratings found after filtering rating > 0.")

        data["user_id"] = data["user_id"].astype(np.int64)
        data["anime_id"] = data["anime_id"].astype(np.int64)
        data["rating"] = data["rating"].astype(np.float32)

        self.user_ids = np.sort(data["user_id"].unique())
        self.item_ids = np.sort(data["anime_id"].unique())
        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.item_to_idx = {item_id: idx for idx, item_id in enumerate(self.item_ids)}

        user_idx = data["user_id"].map(self.user_to_idx).to_numpy()
        item_idx = data["anime_id"].map(self.item_to_idx).to_numpy()
        ratings_values = data["rating"].to_numpy(dtype=np.float32)

        self.global_mean = float(ratings_values.mean())
        centered_values = ratings_values - self.global_mean

        n_users = len(self.user_ids)
        n_items = len(self.item_ids)
        self._user_item_matrix = csr_matrix(
            (centered_values, (user_idx, item_idx)), shape=(n_users, n_items)
        )

        self.item_popularity = np.asarray(self._user_item_matrix.getnnz(axis=0)).ravel()
        self._ratings_by_user = self._build_user_history(data)

        rng = np.random.default_rng(self.config.random_state)
        self.user_factors = rng.normal(scale=0.01, size=(n_users, self.config.n_factors)).astype(
            np.float32
        )
        self.item_factors = rng.normal(scale=0.01, size=(n_items, self.config.n_factors)).astype(
            np.float32
        )

        reg_eye = self.config.reg * np.eye(self.config.n_factors, dtype=np.float32)

        for _ in range(self.config.n_iters):
            self.user_factors = self._least_squares_step(
                self._user_item_matrix, self.item_factors, reg_eye
            )
            self.item_factors = self._least_squares_step(
                self._user_item_matrix.T.tocsr(), self.user_factors, reg_eye
            )

        return self

    def _build_user_history(self, ratings: pd.DataFrame) -> dict[int, set[int]]:
        history: dict[int, set[int]] = {}
        for user_id, group in ratings.groupby("user_id"):
            history[int(user_id)] = set(group["anime_id"].astype(np.int64).tolist())
        return history

    def _least_squares_step(
        self,
        rating_matrix: csr_matrix,
        fixed_factors: np.ndarray,
        reg_eye: np.ndarray,
    ) -> np.ndarray:
        n_rows = rating_matrix.shape[0]
        n_factors = fixed_factors.shape[1]
        updated = np.zeros((n_rows, n_factors), dtype=np.float32)

        for row_idx in range(n_rows):
            start = rating_matrix.indptr[row_idx]
            end = rating_matrix.indptr[row_idx + 1]
            item_indices = rating_matrix.indices[start:end]
            row_values = rating_matrix.data[start:end]

            if item_indices.size == 0:
                continue

            selected_factors = fixed_factors[item_indices]
            a_matrix = selected_factors.T @ selected_factors + reg_eye
            b_vector = selected_factors.T @ row_values
            updated[row_idx] = np.linalg.solve(a_matrix, b_vector)

        return updated

    def predict(self, user_id: int, anime_id: int) -> float:
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model has not been fit yet.")
        if user_id not in self.user_to_idx or anime_id not in self.item_to_idx:
            return self.global_mean

        user_idx = self.user_to_idx[user_id]
        item_idx = self.item_to_idx[anime_id]
        score = float(self.global_mean + self.user_factors[user_idx] @ self.item_factors[item_idx])
        return float(np.clip(score, 1.0, 10.0))

    def _prepare_anime_lookup(self, anime_df: pd.DataFrame) -> pd.DataFrame:
        anime_lookup = anime_df.copy()
        anime_lookup["anime_id"] = pd.to_numeric(anime_lookup["anime_id"], errors="coerce")
        anime_lookup = anime_lookup.dropna(subset=["anime_id"])
        anime_lookup["anime_id"] = anime_lookup["anime_id"].astype(np.int64)
        anime_lookup = anime_lookup.drop_duplicates(subset=["anime_id"])
        return anime_lookup.set_index("anime_id")

    def find_anime_id_by_name(self, anime_df: pd.DataFrame, anime_name: str) -> int:
        anime_lookup = anime_df.copy()
        anime_lookup["anime_id"] = pd.to_numeric(anime_lookup["anime_id"], errors="coerce")
        anime_lookup = anime_lookup.dropna(subset=["anime_id", "name"])
        anime_lookup["anime_id"] = anime_lookup["anime_id"].astype(np.int64)

        normalized_name = anime_name.strip().casefold()
        names = anime_lookup["name"].astype(str).str.casefold()

        exact_match = anime_lookup[names == normalized_name]
        if not exact_match.empty:
            return int(exact_match.iloc[0]["anime_id"])

        partial_match = anime_lookup[names.str.contains(normalized_name, regex=False, na=False)]
        if not partial_match.empty:
            sort_columns = [column for column in ["rating", "members"] if column in partial_match.columns]
            if sort_columns:
                partial_match = partial_match.sort_values(sort_columns, ascending=False)
            return int(partial_match.iloc[0]["anime_id"])

        raise ValueError(f'Could not find anime name matching "{anime_name}".')

    def _als_similarity_scores(self, anime_id: int) -> np.ndarray:
        if self.item_factors is None or self.item_ids is None:
            raise ValueError("Model has not been fit yet.")
        if anime_id not in self.item_to_idx:
            raise ValueError(f"Anime id {anime_id} is not available in the trained ALS model.")

        query_idx = self.item_to_idx[anime_id]
        query_vector = self.item_factors[query_idx]
        norms = np.linalg.norm(self.item_factors, axis=1)
        query_norm = float(np.linalg.norm(query_vector))
        scores = self.item_factors @ query_vector
        scores = scores / (norms * query_norm + 1e-12)
        scores = scores.astype(np.float32)
        scores[query_idx] = -np.inf
        return scores

    def _content_similarity_scores(self, anime_id: int, anime_df: pd.DataFrame) -> np.ndarray:
        if self.item_ids is None:
            raise ValueError("Model has not been fit yet.")
        if anime_id not in self.item_to_idx:
            raise ValueError(f"Anime id {anime_id} is not available in the trained ALS model.")

        content_matrix = self._build_content_matrix(anime_df)
        query_idx = self.item_to_idx[anime_id]
        scores = (content_matrix @ content_matrix[query_idx].T).toarray().ravel()
        scores = scores.astype(np.float32)
        scores[query_idx] = -np.inf
        return scores

    def _build_content_matrix(self, anime_df: pd.DataFrame) -> csr_matrix:
        if self.item_ids is None:
            raise ValueError("Model has not been fit yet.")

        anime_lookup = self._prepare_anime_lookup(anime_df)
        item_frame = anime_lookup.reindex(self.item_ids).reset_index()

        genre_text = item_frame.get("genre", pd.Series("", index=item_frame.index))
        genre_text = genre_text.fillna("").astype(str).str.replace(",", " ", regex=False)
        if genre_text.str.strip().any():
            genre_features = TfidfVectorizer(min_df=1).fit_transform(genre_text)
        else:
            genre_features = csr_matrix((len(item_frame), 0), dtype=np.float32)

        type_series = item_frame.get("type", pd.Series("Unknown", index=item_frame.index))
        type_dummies = pd.get_dummies(type_series.fillna("Unknown").astype(str), dtype=np.float32)
        type_features = csr_matrix(type_dummies.to_numpy(dtype=np.float32))

        numeric_columns = [column for column in ["rating", "members"] if column in item_frame.columns]
        if numeric_columns:
            numeric = item_frame[numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            numeric_values = numeric.to_numpy(dtype=np.float32)
            mins = numeric_values.min(axis=0)
            ranges = numeric_values.max(axis=0) - mins
            ranges[ranges == 0] = 1.0
            numeric_values = (numeric_values - mins) / ranges
            numeric_features = csr_matrix(numeric_values)
        else:
            numeric_features = csr_matrix((len(item_frame), 0), dtype=np.float32)

        return normalize(hstack([genre_features, type_features, numeric_features]).tocsr())

    def _top_indices(self, scores: np.ndarray, top_k: int) -> np.ndarray:
        if top_k >= len(scores):
            return np.argsort(-scores)

        top_indices = np.argpartition(-scores, top_k)[:top_k]
        return top_indices[np.argsort(-scores[top_indices])]

    def evaluate(self, test_ratings: pd.DataFrame) -> dict[str, float]:
        """
        Evaluate model on test ratings using RMSE and MAE.
        """
        required_columns = {"user_id", "anime_id", "rating"}
        missing_columns = required_columns - set(test_ratings.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

        data = test_ratings[["user_id", "anime_id", "rating"]].copy()
        data["rating"] = pd.to_numeric(data["rating"], errors="coerce")
        data = data.dropna(subset=["user_id", "anime_id", "rating"])
        data = data[data["rating"] > 0].copy()

        if data.empty:
            raise ValueError("No valid positive ratings found for evaluation.")

        y_true = []
        y_pred = []

        for row in data.itertuples(index=False):
            user_id = int(row.user_id)
            anime_id = int(row.anime_id)
            true_rating = float(row.rating)

            predicted_rating = self.predict(user_id, anime_id)

            y_true.append(true_rating)
            y_pred.append(predicted_rating)

        y_true = np.array(y_true, dtype=np.float32)
        y_pred = np.array(y_pred, dtype=np.float32)

        errors = y_true - y_pred

        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(np.abs(errors)))

        return {
            "rmse": rmse,
            "mae": mae,
            "n_test": int(len(y_true)),
        }

    def similar_anime(
        self,
        anime_id: int,
        anime_df: pd.DataFrame,
        top_k: int = 5,
    ) -> pd.DataFrame:
        anime_lookup = self._prepare_anime_lookup(anime_df)
        similarity = self._als_similarity_scores(anime_id)
        top_indices = self._top_indices(similarity, top_k)

        result = pd.DataFrame(
            {
                "anime_id": self.item_ids[top_indices],
                "similarity": similarity[top_indices],
            }
        )
        result = result.join(anime_lookup[["name", "genre", "type"]], on="anime_id")
        return result.reset_index(drop=True)

    def hybrid_similar_anime(
        self,
        anime_id: int,
        anime_df: pd.DataFrame,
        top_k: int = 5,
        content_weight: float = 0.5,
    ) -> pd.DataFrame:
        if self.item_ids is None:
            raise ValueError("Model has not been fit yet.")

        content_weight = float(np.clip(content_weight, 0.0, 1.0))
        als_weight = 1.0 - content_weight

        anime_lookup = self._prepare_anime_lookup(anime_df)
        als_scores = self._als_similarity_scores(anime_id)
        content_scores = self._content_similarity_scores(anime_id, anime_df)
        hybrid_scores = als_weight * als_scores + content_weight * content_scores
        hybrid_scores[self.item_to_idx[anime_id]] = -np.inf

        top_indices = self._top_indices(hybrid_scores, top_k)
        result = pd.DataFrame(
            {
                "anime_id": self.item_ids[top_indices],
                "hybrid_score": hybrid_scores[top_indices],
                "als_similarity": als_scores[top_indices],
                "content_similarity": content_scores[top_indices],
            }
        )
        result = result.join(anime_lookup[["name", "genre", "type"]], on="anime_id")
        return result.reset_index(drop=True)

    def recommend(
        self,
        user_id: int,
        anime_df: pd.DataFrame,
        top_k: int = 10,
        exclude_seen: bool = True,
    ) -> pd.DataFrame:
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model has not been fit yet.")

        anime_lookup = self._prepare_anime_lookup(anime_df)

        if user_id in self.user_to_idx:
            user_idx = self.user_to_idx[user_id]
            scores = self.global_mean + self.item_factors @ self.user_factors[user_idx]
            scores = scores.astype(np.float32)

            if exclude_seen:
                seen_ids = self._ratings_by_user.get(user_id, set())
                seen_mask = np.array([item_id in seen_ids for item_id in self.item_ids])
                scores = scores.copy()
                scores[seen_mask] = -np.inf
        else:
            if self.item_popularity is None:
                scores = np.zeros(len(self.item_ids), dtype=np.float32)
            else:
                scores = self.item_popularity.astype(np.float32)

        if top_k >= len(scores):
            top_indices = np.argsort(-scores)
        else:
            top_indices = np.argpartition(-scores, top_k)[:top_k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]

        result = pd.DataFrame(
            {
                "anime_id": self.item_ids[top_indices],
                "predicted_rating": np.clip(scores[top_indices], 1.0, 10.0),
            }
        )

        result = result.join(anime_lookup[["name", "genre", "type"]], on="anime_id")
        result = result.reset_index(drop=True)
        return result

    def save(self, directory: str | Path) -> None:
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model has not been fit yet.")

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config,
            "global_mean": self.global_mean,
            "user_factors": self.user_factors,
            "item_factors": self.item_factors,
            "user_ids": self.user_ids,
            "item_ids": self.item_ids,
            "user_to_idx": self.user_to_idx,
            "item_to_idx": self.item_to_idx,
            "item_popularity": self.item_popularity,
            "ratings_by_user": self._ratings_by_user,
        }
        with open(directory / "als_model.pkl", "wb") as handle:
            pickle.dump(payload, handle)

        with open(directory / "als_metadata.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "n_users": int(len(self.user_ids) if self.user_ids is not None else 0),
                    "n_items": int(len(self.item_ids) if self.item_ids is not None else 0),
                    "n_factors": int(self.config.n_factors),
                    "n_iters": int(self.config.n_iters),
                    "reg": float(self.config.reg),
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, directory: str | Path) -> "ExplicitALSRecommender":
        directory = Path(directory)
        with open(directory / "als_model.pkl", "rb") as handle:
            payload = pickle.load(handle)

        model = cls(payload["config"])
        model.global_mean = payload["global_mean"]
        model.user_factors = payload["user_factors"]
        model.item_factors = payload["item_factors"]
        model.user_ids = payload["user_ids"]
        model.item_ids = payload["item_ids"]
        model.user_to_idx = payload["user_to_idx"]
        model.item_to_idx = payload["item_to_idx"]
        model.item_popularity = payload.get("item_popularity")
        model._ratings_by_user = payload.get("ratings_by_user", {})
        return model


def load_data(
    ratings_path: str | Path,
    anime_path: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(ratings_path)
    anime = pd.read_csv(anime_path)

    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["user_id", "anime_id", "rating"])
    ratings = ratings[ratings["rating"] > 0].copy()

    train_ratings, test_ratings = train_test_split(
        ratings,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    return train_ratings, test_ratings, anime



def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Train an explicit ALS anime recommender.")
    parser.add_argument(
        "--ratings",
        type=str,
        default=str(Path("anime data") / "rating.csv"),
        help="Path to rating.csv",
    )
    parser.add_argument(
        "--anime",
        type=str,
        default=str(Path("anime data") / "anime.csv"),
        help="Path to anime.csv",
    )
    parser.add_argument("--n-factors", type=int, default=32)
    parser.add_argument("--n-iters", type=int, default=15)
    parser.add_argument("--reg", type=float, default=0.5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.01)
    parser.add_argument("--user-id", type=int, default=1, help="User id to print recommendations for.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--anime-id", type=int, default=None, help="Anime id to find similar titles for.")
    parser.add_argument("--anime-name", type=str, default=None, help="Anime name to find similar titles for.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["als", "hybrid"],
        default="hybrid",
        help="Similarity mode used when --anime-id or --anime-name is provided.",
    )
    parser.add_argument(
        "--content-weight",
        type=float,
        default=0.5,
        help="Hybrid weight for content similarity. 0 means ALS only, 1 means content only.",
    )
    parser.add_argument("--save-dir", type=str, default="artifacts/als_model")
    args = parser.parse_args()

    train_df, test_df, anime_df = load_data(
    args.ratings,
    args.anime,
    test_size=args.test_size,
    random_state=args.random_state,
    )

    config = ALSConfig(
        n_factors=args.n_factors,
        n_iters=args.n_iters,
        reg=args.reg,
        random_state=args.random_state,
    )
    model = ExplicitALSRecommender(config)
    model.fit(train_df)

    metrics = model.evaluate(test_df)

    print("Evaluation on test set:")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE : {metrics['mae']:.4f}")
    print(f"Test samples: {metrics['n_test']}")
    print()

    if args.anime_id is not None or args.anime_name is not None:
        if args.anime_id is not None:
            query_anime_id = args.anime_id
        else:
            query_anime_id = model.find_anime_id_by_name(anime_df, args.anime_name or "")

        if args.mode == "hybrid":
            similar_titles = model.hybrid_similar_anime(
                query_anime_id,
                anime_df,
                top_k=args.top_k,
                content_weight=args.content_weight,
            )
        else:
            similar_titles = model.similar_anime(query_anime_id, anime_df, top_k=args.top_k)

        query_row = anime_df[pd.to_numeric(anime_df["anime_id"], errors="coerce") == query_anime_id]
        query_name = query_row.iloc[0]["name"] if not query_row.empty else str(query_anime_id)
        print(f"Top {args.top_k} similar anime for: {query_name} (anime_id={query_anime_id})")
        print(f"mode={args.mode}, content_weight={args.content_weight:.2f}")
        print(similar_titles.to_string(index=False))
    else:
        recommendations = model.recommend(args.user_id, anime_df, top_k=args.top_k)
        print(f"Top {args.top_k} recommendations for user {args.user_id}:")
        print(recommendations.to_string(index=False))

    model.save(args.save_dir)
    print(f"\nModel saved to: {Path(args.save_dir) / 'als_model.pkl'}")


if __name__ == "__main__":
    main()
