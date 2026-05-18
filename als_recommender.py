from __future__ import annotations

import argparse
import html
import json
import pickle
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


@dataclass
class ALSConfig:
    n_factors: int = 16
    n_iters: int = 5
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

    def evaluate(self, ratings: pd.DataFrame) -> dict[str, float | int]:
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model has not been fit yet.")

        test_data = clean_explicit_ratings(ratings)
        if test_data.empty:
            raise ValueError("No positive ratings found for evaluation.")

        predictions = np.array(
            [
                self.predict(int(row.user_id), int(row.anime_id))
                for row in test_data.itertuples(index=False)
            ],
            dtype=np.float32,
        )
        actual = test_data["rating"].to_numpy(dtype=np.float32)
        errors = predictions - actual

        known_mask = (
            test_data["user_id"].isin(self.user_to_idx)
            & test_data["anime_id"].isin(self.item_to_idx)
        )
        baseline_errors = np.full(len(actual), self.global_mean, dtype=np.float32) - actual

        return {
            "n_test": int(len(test_data)),
            "coverage": float(known_mask.mean()),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "mae": float(np.mean(np.abs(errors))),
            "baseline_rmse": float(np.sqrt(np.mean(baseline_errors**2))),
            "baseline_mae": float(np.mean(np.abs(baseline_errors))),
        }

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

    def _normalize_title(self, title: object) -> str:
        if pd.isna(title):
            return ""

        text = html.unescape(str(title)).casefold()
        text = re.sub(r"\([^)]*\)", " ", text)
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _shared_prefix_ratio(self, query_title: str, candidate_title: str) -> float:
        query_tokens = query_title.split()
        candidate_tokens = candidate_title.split()
        if not query_tokens or not candidate_tokens:
            return 0.0

        prefix_len = 0
        for query_token, candidate_token in zip(query_tokens, candidate_tokens):
            if query_token != candidate_token:
                break
            prefix_len += 1

        min_len = min(len(query_tokens), len(candidate_tokens))
        if prefix_len < min(2, min_len):
            return 0.0

        return prefix_len / max(len(query_tokens), len(candidate_tokens))

    def _title_similarity_score_pair(self, query_title: str, candidate_title: str) -> float:
        if not query_title or not candidate_title:
            return 0.0

        if candidate_title == query_title:
            return 1.0
        if candidate_title.startswith(f"{query_title} ") or query_title.startswith(f"{candidate_title} "):
            return 0.95

        shared_prefix_score = self._shared_prefix_ratio(query_title, candidate_title)
        if shared_prefix_score > 0:
            return 0.85 + 0.1 * shared_prefix_score

        shorter_token_count = min(len(query_title.split()), len(candidate_title.split()))
        if shorter_token_count >= 2 and (
            f" {query_title} " in f" {candidate_title} "
            or f" {candidate_title} " in f" {query_title} "
        ):
            return 0.9

        query_tokens = set(query_title.split())
        candidate_tokens = set(candidate_title.split())
        token_score = (
            len(query_tokens & candidate_tokens) / len(query_tokens | candidate_tokens)
            if query_tokens and candidate_tokens
            else 0.0
        )
        return float(token_score if token_score >= 0.5 else 0.0)

    def _title_similarity_scores(self, anime_id: int, anime_df: pd.DataFrame) -> np.ndarray:
        if self.item_ids is None:
            raise ValueError("Model has not been fit yet.")
        if anime_id not in self.item_to_idx:
            raise ValueError(f"Anime id {anime_id} is not available in the trained ALS model.")

        anime_lookup = self._prepare_anime_lookup(anime_df)
        item_frame = anime_lookup.reindex(self.item_ids)
        query_name = item_frame.loc[anime_id, "name"] if anime_id in item_frame.index else ""
        query_title = self._normalize_title(query_name)
        scores = np.zeros(len(item_frame), dtype=np.float32)

        if not query_title:
            return scores

        for idx, candidate_name in enumerate(item_frame.get("name", pd.Series("", index=item_frame.index))):
            candidate_title = self._normalize_title(candidate_name)
            if not candidate_title:
                continue

            scores[idx] = self._title_similarity_score_pair(query_title, candidate_title)

        scores[self.item_to_idx[anime_id]] = -np.inf
        return scores

    def _metadata_title_candidates(
        self,
        anime_id: int,
        anime_df: pd.DataFrame,
        min_title_score: float = 0.85,
    ) -> pd.DataFrame:
        if self.item_ids is None:
            raise ValueError("Model has not been fit yet.")

        anime_lookup = self._prepare_anime_lookup(anime_df)
        if anime_id not in anime_lookup.index:
            return pd.DataFrame()

        query_title = self._normalize_title(anime_lookup.loc[anime_id, "name"])
        trained_ids = set(int(item_id) for item_id in self.item_ids.tolist())
        rows = []

        for candidate_id, candidate_row in anime_lookup.iterrows():
            candidate_id = int(candidate_id)
            if candidate_id == anime_id or candidate_id in trained_ids:
                continue

            candidate_title = self._normalize_title(candidate_row.get("name", ""))
            title_score = self._title_similarity_score_pair(query_title, candidate_title)
            if title_score < min_title_score:
                continue

            rows.append(
                {
                    "anime_id": candidate_id,
                    "hybrid_score": title_score, #để so sánh với hybrid score của các mục đã được đào tạo
                    "als_similarity": np.nan,
                    "content_similarity": np.nan,
                    "title_similarity": title_score,
                    "name": candidate_row.get("name"),
                    "genre": candidate_row.get("genre"),
                    "type": candidate_row.get("type"),
                }
            )

        return pd.DataFrame(rows)

    def _top_indices(self, scores: np.ndarray, top_k: int) -> np.ndarray:
        if top_k >= len(scores):
            return np.argsort(-scores)

        top_indices = np.argpartition(-scores, top_k)[:top_k]
        return top_indices[np.argsort(-scores[top_indices])]

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

    # Kết hợp điểm tương đồng ALS, nội dung và tiêu đề để tìm anime tương tự
    def hybrid_similar_anime(
        self,
        anime_id: int,
        anime_df: pd.DataFrame,
        top_k: int = 5,
        content_weight: float = 0.5,
        title_weight: float = 0.45,
    ) -> pd.DataFrame:
        if self.item_ids is None:
            raise ValueError("Model has not been fit yet.")

        content_weight = float(np.clip(content_weight, 0.0, 1.0))
        title_weight = float(np.clip(title_weight, 0.0, 1.0))
        base_weight = 1.0 - title_weight
        als_weight = base_weight * (1.0 - content_weight)
        content_weight = base_weight * content_weight

        anime_lookup = self._prepare_anime_lookup(anime_df)
        als_scores = self._als_similarity_scores(anime_id)
        content_scores = self._content_similarity_scores(anime_id, anime_df)
        title_scores = self._title_similarity_scores(anime_id, anime_df)
        hybrid_scores = als_weight * als_scores + content_weight * content_scores + title_weight * title_scores
        hybrid_scores[self.item_to_idx[anime_id]] = -np.inf

        top_indices = self._top_indices(hybrid_scores, top_k)
        result = pd.DataFrame(
            {
                "anime_id": self.item_ids[top_indices],
                "hybrid_score": hybrid_scores[top_indices],
                "als_similarity": als_scores[top_indices],
                "content_similarity": content_scores[top_indices],
                "title_similarity": title_scores[top_indices],
            }
        )
        result = result.join(anime_lookup[["name", "genre", "type"]], on="anime_id")
        metadata_title_candidates = self._metadata_title_candidates(anime_id, anime_df)
        if not metadata_title_candidates.empty:
            result = pd.concat([result.reset_index(drop=True), metadata_title_candidates], ignore_index=True)
            result = result.sort_values("hybrid_score", ascending=False).head(top_k)

        return result.reset_index(drop=True)

    # đề xuất anime dựa trên người dùng đã xem và đánh giá
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

    def save(
        self,
        directory: str | Path,
        model_filename: str = "als_model.pkl",
        metadata_extra: dict[str, object] | None = None,
    ) -> None:
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model has not been fit yet.")

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
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
        with open(directory / model_filename, "wb") as handle:
            pickle.dump(payload, handle)

        metadata = {
            "n_users": int(len(self.user_ids) if self.user_ids is not None else 0),
            "n_items": int(len(self.item_ids) if self.item_ids is not None else 0),
            "n_factors": int(self.config.n_factors),
            "n_iters": int(self.config.n_iters),
            "reg": float(self.config.reg),
            "model_filename": model_filename,
        }
        if metadata_extra:
            metadata.update(metadata_extra)

        with open(directory / "als_metadata.json", "w", encoding="utf-8") as handle:
            json.dump(
                metadata,
                handle,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, directory: str | Path, model_filename: str = "als_model.pkl") -> "ExplicitALSRecommender":
        directory = Path(directory)
        with open(directory / model_filename, "rb") as handle:
            payload = pickle.load(handle)

        config_payload = payload["config"]
        config = ALSConfig(**config_payload) if isinstance(config_payload, dict) else config_payload
        model = cls(config)
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


def load_data(ratings_path: str | Path, anime_path: str | Path, max_rows: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(ratings_path)
    anime = pd.read_csv(anime_path)

    if max_rows is not None and max_rows > 0:
        ratings = ratings.head(max_rows).copy()

    return ratings, anime


def clean_explicit_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"user_id", "anime_id", "rating"}
    missing_columns = required_columns - set(ratings.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    data = ratings[["user_id", "anime_id", "rating"]].copy()
    data["user_id"] = pd.to_numeric(data["user_id"], errors="coerce")
    data["anime_id"] = pd.to_numeric(data["anime_id"], errors="coerce")
    data["rating"] = pd.to_numeric(data["rating"], errors="coerce")
    data = data.dropna(subset=["user_id", "anime_id", "rating"])
    data = data[data["rating"] > 0].copy()
    data["user_id"] = data["user_id"].astype(np.int64)
    data["anime_id"] = data["anime_id"].astype(np.int64)
    data["rating"] = data["rating"].astype(np.float32)
    return data


def train_test_split_by_user(
    ratings: pd.DataFrame,
    test_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, _, test = train_dev_test_split_by_user(
        ratings,
        dev_size=0.0,
        test_size=test_size,
        random_state=random_state,
    )
    return train, test


def train_dev_test_split_by_user(
    ratings: pd.DataFrame,
    dev_size: float = 0.05,
    test_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = clean_explicit_ratings(ratings)
    if data.empty:
        raise ValueError("No positive ratings found after filtering rating > 0.")

    rng = np.random.default_rng(random_state)
    train_parts: list[pd.DataFrame] = []
    dev_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    dev_size = float(np.clip(dev_size, 0.0, 0.8))
    test_size = float(np.clip(test_size, 0.01, 0.9))
    if dev_size + test_size >= 0.95:
        raise ValueError("dev_size + test_size must be less than 0.95 so train data is not empty.")

    for _, group in data.groupby("user_id", sort=False):
        if len(group) < 2:
            train_parts.append(group)
            continue

        n_test = int(round(len(group) * test_size))
        n_test = min(max(n_test, 1), len(group) - 1)
        test_index = rng.choice(group.index.to_numpy(), size=n_test, replace=False)
        test_mask = group.index.isin(test_index)
        test_parts.append(group.loc[test_mask])

        remaining = group.loc[~test_mask]
        if dev_size > 0 and len(remaining) >= 2:
            n_dev = int(round(len(group) * dev_size))
            n_dev = min(max(n_dev, 1), len(remaining) - 1)
            dev_index = rng.choice(remaining.index.to_numpy(), size=n_dev, replace=False)
            dev_mask = remaining.index.isin(dev_index)
            dev_parts.append(remaining.loc[dev_mask])
            train_parts.append(remaining.loc[~dev_mask])
        else:
            train_parts.append(remaining)

    train = pd.concat(train_parts, ignore_index=True) if train_parts else data.iloc[0:0].copy()
    dev = pd.concat(dev_parts, ignore_index=True) if dev_parts else data.iloc[0:0].copy()
    test = pd.concat(test_parts, ignore_index=True) if test_parts else data.iloc[0:0].copy()
    if test.empty:
        raise ValueError("Not enough users with at least 2 ratings to create an evaluation split.")

    return train, dev, test


def evaluate_als_model(
    ratings: pd.DataFrame,
    config: ALSConfig | None = None,
    dev_size: float = 0.05,
    test_size: float = 0.1,
    random_state: int = 42,
) -> tuple[
    ExplicitALSRecommender,
    dict[str, dict[str, float | int]],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    train_ratings, dev_ratings, test_ratings = train_dev_test_split_by_user(
        ratings,
        dev_size=dev_size,
        test_size=test_size,
        random_state=random_state,
    )
    model = ExplicitALSRecommender(config)
    model.fit(train_ratings)
    metrics = {"test": model.evaluate(test_ratings)}
    if not dev_ratings.empty:
        metrics["dev"] = model.evaluate(dev_ratings)
    metrics["split"] = {
        "n_train": int(len(train_ratings)),
        "n_dev": int(len(dev_ratings)),
        "n_test": int(len(test_ratings)),
    }
    return model, metrics, train_ratings, dev_ratings, test_ratings


def parse_int_grid(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("Grid must contain at least one integer value.")
    if any(item <= 0 for item in values):
        raise ValueError("Grid values must be positive integers.")
    return values


def parse_float_grid(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("Grid must contain at least one float value.")
    if any(item <= 0 for item in values):
        raise ValueError("Grid values must be positive floats.")
    return values


def tune_als_model(
    ratings: pd.DataFrame,
    n_factors_grid: list[int],
    n_iters_grid: list[int],
    reg_grid: list[float],
    dev_size: float = 0.05,
    test_size: float = 0.1,
    random_state: int = 42,
) -> tuple[
    ExplicitALSRecommender,
    dict[str, dict[str, float | int] | list[dict[str, float | int]]],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    train_ratings, dev_ratings, test_ratings = train_dev_test_split_by_user(
        ratings,
        dev_size=dev_size,
        test_size=test_size,
        random_state=random_state,
    )
    if dev_ratings.empty:
        raise ValueError("Tuning requires a non-empty dev split. Use --dev-size greater than 0.")

    best_model: ExplicitALSRecommender | None = None
    best_dev_metrics: dict[str, float | int] | None = None
    best_config: ALSConfig | None = None
    tuning_results: list[dict[str, float | int]] = []

    for n_factors in n_factors_grid:
        for n_iters in n_iters_grid:
            for reg in reg_grid:
                candidate_config = ALSConfig(
                    n_factors=n_factors,
                    n_iters=n_iters,
                    reg=reg,
                    random_state=random_state,
                )
                candidate_model = ExplicitALSRecommender(candidate_config)
                candidate_model.fit(train_ratings)
                dev_metrics = candidate_model.evaluate(dev_ratings)
                result = {
                    "n_factors": int(n_factors),
                    "n_iters": int(n_iters),
                    "reg": float(reg),
                    "dev_rmse": float(dev_metrics["rmse"]),
                    "dev_mae": float(dev_metrics["mae"]),
                    "dev_coverage": float(dev_metrics["coverage"]),
                }
                tuning_results.append(result)
                print(
                    "tune "
                    f"n_factors={n_factors}, n_iters={n_iters}, reg={reg:.4f} "
                    f"=> dev RMSE={dev_metrics['rmse']:.4f}, MAE={dev_metrics['mae']:.4f}"
                )

                if best_dev_metrics is None or dev_metrics["rmse"] < best_dev_metrics["rmse"]:
                    best_model = candidate_model
                    best_dev_metrics = dev_metrics
                    best_config = candidate_config

    if best_model is None or best_dev_metrics is None or best_config is None:
        raise ValueError("No ALS tuning candidates were evaluated.")

    metrics: dict[str, dict[str, float | int] | list[dict[str, float | int]]] = {
        "dev": best_dev_metrics,
        "test": best_model.evaluate(test_ratings),
        "split": {
            "n_train": int(len(train_ratings)),
            "n_dev": int(len(dev_ratings)),
            "n_test": int(len(test_ratings)),
        },
        "best_config": asdict(best_config),
        "tuning_results": tuning_results,
    }
    return best_model, metrics, train_ratings, dev_ratings, test_ratings



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
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None, help="Maximum rating rows to load. Default uses all rows.")
    parser.add_argument("--user-id", type=int, default=1, help="User id to print recommendations for.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--anime-id", type=int, default=None, help="Anime id to find similar titles for.")
    parser.add_argument("--anime-name", type=str, default=None, help="Anime name to find similar titles for.")
    parser.add_argument(
        "--content-weight",
        type=float,
        default=0.5,
        help="Hybrid weight for content similarity. 0 means ALS only, 1 means content only.",
    )
    parser.add_argument(
        "--n-factors-grid",
        type=str,
        default="16,32,64",
        help="Comma-separated ALS factor values used for tuning.",
    )
    parser.add_argument(
        "--n-iters-grid",
        type=str,
        default="5,8,10",
        help="Comma-separated ALS iteration values used for tuning.",
    )
    parser.add_argument(
        "--reg-grid",
        type=str,
        default="0.05,0.1,0.2",
        help="Comma-separated regularization values used for tuning.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.05,
        help="Fraction of each user's ratings held out as a test set during tuning.",
    )
    parser.add_argument(
        "--dev-size",
        type=float,
        default=0.05,
        help="Fraction of each user's ratings held out as a dev set during tuning.",
    )
    parser.add_argument(
        "--title-weight",
        type=float,
        default=0.45,
        help="Hybrid weight for strict title/franchise similarity.",
    )
    parser.add_argument("--no-save", action="store_true", help="Do not save the trained model artifacts.")
    parser.add_argument("--save-dir", type=str, default="artifacts/als_model")
    parser.add_argument("--model-filename", type=str, default="als_model.pkl")
    args = parser.parse_args()

    ratings_df, anime_df = load_data(args.ratings, args.anime, max_rows=args.max_rows)
    split_metadata: dict[str, object] = {
        "trained_on": "all_loaded_ratings",
        "max_rows": args.max_rows,
    }

    n_factors_grid = parse_int_grid(args.n_factors_grid)
    n_iters_grid = parse_int_grid(args.n_iters_grid)
    reg_grid = parse_float_grid(args.reg_grid)
    print("Tuning ALS hyperparameters")
    print(f"n_factors grid: {n_factors_grid}")
    print(f"n_iters grid: {n_iters_grid}")
    print(f"reg grid: {reg_grid}")
    print()

    model, metrics, _, _, _ = tune_als_model(
        ratings_df,
        n_factors_grid=n_factors_grid,
        n_iters_grid=n_iters_grid,
        reg_grid=reg_grid,
        dev_size=args.dev_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    best_config = metrics["best_config"]
    print("\nBest ALS config")
    print(
        f"n_factors={best_config['n_factors']}, "
        f"n_iters={best_config['n_iters']}, "
        f"reg={best_config['reg']}"
    )

    print("\nDev metrics")
    dev_metrics = metrics["dev"]
    print(f"known user/item coverage: {dev_metrics['coverage']:.3f}")
    print(f"RMSE: {dev_metrics['rmse']:.4f}  | baseline RMSE: {dev_metrics['baseline_rmse']:.4f}")
    print(f"MAE : {dev_metrics['mae']:.4f}  | baseline MAE : {dev_metrics['baseline_mae']:.4f}")

    test_metrics = metrics["test"]
    print("\nTest metrics")
    print(f"known user/item coverage: {test_metrics['coverage']:.3f}")
    print(f"RMSE: {test_metrics['rmse']:.4f}  | baseline RMSE: {test_metrics['baseline_rmse']:.4f}")
    print(f"MAE : {test_metrics['mae']:.4f}  | baseline MAE : {test_metrics['baseline_mae']:.4f}")
    print()
    split_metadata = {
        "trained_on": "train_split_only_tuned",
        "split": metrics["split"],
        "dev_size": args.dev_size,
        "test_size": args.test_size,
        "max_rows": args.max_rows,
        "best_config": metrics["best_config"],
        "dev_metrics": metrics["dev"],
        "test_metrics": metrics["test"],
        "tuning_results": metrics["tuning_results"],
    }

    if args.anime_id is not None or args.anime_name is not None:
        if args.anime_id is not None:
            query_anime_id = args.anime_id
        else:
            query_anime_id = model.find_anime_id_by_name(anime_df, args.anime_name or "")

        similar_titles = model.hybrid_similar_anime(
            query_anime_id,
            anime_df,
            top_k=args.top_k,
            content_weight=args.content_weight,
            title_weight=args.title_weight,
        )

        query_row = anime_df[pd.to_numeric(anime_df["anime_id"], errors="coerce") == query_anime_id]
        query_name = query_row.iloc[0]["name"] if not query_row.empty else str(query_anime_id)
        print(f"Top {args.top_k} similar anime for: {query_name} (anime_id={query_anime_id})")
        print(f"mode=hybrid, content_weight={args.content_weight:.2f}, title_weight={args.title_weight:.2f}")
        print(similar_titles.to_string(index=False))
    else:
        recommendations = model.recommend(args.user_id, anime_df, top_k=args.top_k)
        print(f"Top {args.top_k} recommendations for user {args.user_id}:")
        print(recommendations.to_string(index=False))

    if not args.no_save:
        model.save(args.save_dir, model_filename=args.model_filename, metadata_extra=split_metadata)
        print(f"\nModel saved to: {Path(args.save_dir) / args.model_filename}")


if __name__ == "__main__":
    main()
