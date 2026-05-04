from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


DATA_DIR = Path("anime data")


@dataclass
class PreparedData:
    anime: pd.DataFrame
    interactions: pd.DataFrame
    user_to_idx: dict[int, int]
    item_to_idx: dict[int, int]
    idx_to_item: dict[int, int]
    user_positive_items: list[set[int]]


def load_data(max_ratings: int | None = 200_000, min_rating: int = 7) -> PreparedData:
    """Load ratings and convert explicit ratings into implicit positive feedback."""
    anime = pd.read_csv(DATA_DIR / "anime.csv").dropna(subset=["name"]).copy()
    ratings = pd.read_csv(DATA_DIR / "rating.csv")

    if max_ratings is not None:
        ratings = ratings.head(max_ratings)

    ratings = ratings[ratings["rating"] >= min_rating].copy()
    valid_anime_ids = set(anime["anime_id"])
    ratings = ratings[ratings["anime_id"].isin(valid_anime_ids)]

    user_ids = ratings["user_id"].unique()
    item_ids = ratings["anime_id"].unique()

    user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_to_idx = {anime_id: idx for idx, anime_id in enumerate(item_ids)}
    idx_to_item = {idx: anime_id for anime_id, idx in item_to_idx.items()}

    interactions = pd.DataFrame(
        {
            "user_idx": ratings["user_id"].map(user_to_idx),
            "item_idx": ratings["anime_id"].map(item_to_idx),
        }
    ).drop_duplicates()

    user_positive_items = [set() for _ in range(len(user_to_idx))]
    for user_idx, item_idx in interactions[["user_idx", "item_idx"]].to_numpy():
        user_positive_items[int(user_idx)].add(int(item_idx))

    return PreparedData(
        anime=anime,
        interactions=interactions,
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        idx_to_item=idx_to_item,
        user_positive_items=user_positive_items,
    )


class BPRRecommender:
    """Small BPR implementation intended for learning the algorithm syntax."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        factors: int = 32,
        learning_rate: float = 0.05,
        regularization: float = 0.002,
        epochs: int = 10,
        random_state: int = 42,
    ) -> None:
        self.n_users = n_users
        self.n_items = n_items
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.rng = np.random.default_rng(random_state)

        scale = 0.01
        self.user_factors = self.rng.normal(0, scale, size=(n_users, factors))
        self.item_factors = self.rng.normal(0, scale, size=(n_items, factors))

    def fit(self, user_positive_items: list[set[int]]) -> "BPRRecommender":
        train_users = [u for u, positives in enumerate(user_positive_items) if positives]

        for epoch in range(self.epochs):
            self.rng.shuffle(train_users)
            for user_idx in train_users:
                positive_item = self.rng.choice(list(user_positive_items[user_idx]))
                negative_item = self._sample_negative(user_positive_items[user_idx])
                self._update(user_idx, int(positive_item), negative_item)

            print(f"epoch {epoch + 1}/{self.epochs} finished")

        return self

    def _sample_negative(self, user_positives: set[int]) -> int:
        while True:
            item_idx = int(self.rng.integers(0, self.n_items))
            if item_idx not in user_positives:
                return item_idx

    def _update(self, user_idx: int, positive_item: int, negative_item: int) -> None:
        user_vector = self.user_factors[user_idx].copy()
        positive_vector = self.item_factors[positive_item].copy()
        negative_vector = self.item_factors[negative_item].copy()

        x_uij = user_vector @ (positive_vector - negative_vector)
        sigmoid = 1.0 / (1.0 + np.exp(x_uij))

        lr = self.learning_rate
        reg = self.regularization
        self.user_factors[user_idx] += lr * (
            sigmoid * (positive_vector - negative_vector) - reg * user_vector
        )
        self.item_factors[positive_item] += lr * (sigmoid * user_vector - reg * positive_vector)
        self.item_factors[negative_item] += lr * (-sigmoid * user_vector - reg * negative_vector)

    def recommend(
        self,
        user_idx: int,
        known_items: set[int],
        top_k: int = 10,
        allowed_items: np.ndarray | None = None,
    ) -> list[tuple[int, float]]:
        scores = self.user_factors[user_idx] @ self.item_factors.T
        scores[list(known_items)] = -np.inf

        if allowed_items is not None:
            mask = np.full(self.n_items, -np.inf)
            mask[allowed_items] = scores[allowed_items]
            scores = mask

        top_items = np.argpartition(scores, -top_k)[-top_k:]
        top_items = top_items[np.argsort(scores[top_items])[::-1]]
        return [(int(item_idx), float(scores[item_idx])) for item_idx in top_items]


def train_test_split_by_user(
    user_positive_items: list[set[int]], random_state: int = 42
) -> tuple[list[set[int]], dict[int, int]]:
    """Hold out one positive item per user for recall@k evaluation."""
    rng = np.random.default_rng(random_state)
    train = []
    test = {}

    for user_idx, positives in enumerate(user_positive_items):
        if len(positives) < 2:
            train.append(set(positives))
            continue

        held_out = int(rng.choice(list(positives)))
        train.append(set(positives) - {held_out})
        test[user_idx] = held_out

    return train, test


def recall_at_k(model: BPRRecommender, train_items: list[set[int]], test_items: dict[int, int], k: int) -> float:
    hits = 0
    for user_idx, true_item in test_items.items():
        recommended = model.recommend(user_idx, train_items[user_idx], top_k=k)
        recommended_items = {item_idx for item_idx, _ in recommended}
        hits += int(true_item in recommended_items)
    return hits / max(len(test_items), 1)


def build_content_matrix(anime: pd.DataFrame, item_to_idx: dict[int, int]) -> csr_matrix:
    """Create item features from genre, type, rating, and members."""
    item_frame = anime[anime["anime_id"].isin(item_to_idx)].copy()
    item_frame["item_idx"] = item_frame["anime_id"].map(item_to_idx)
    item_frame = item_frame.sort_values("item_idx")

    genre_text = item_frame["genre"].fillna("").str.replace(",", " ", regex=False)
    genre_features = TfidfVectorizer(min_df=2).fit_transform(genre_text)

    type_encoder = OneHotEncoder(handle_unknown="ignore")
    type_features = type_encoder.fit_transform(item_frame[["type"]].fillna("Unknown"))

    numeric = item_frame[["rating", "members"]].fillna(0)
    numeric_features = csr_matrix(MinMaxScaler().fit_transform(numeric))

    return hstack([genre_features, type_features, numeric_features]).tocsr()


def similar_anime_by_content(
    anime_id: int,
    anime: pd.DataFrame,
    item_to_idx: dict[int, int],
    idx_to_item: dict[int, int],
    content_matrix: csr_matrix,
    top_k: int = 10,
) -> pd.DataFrame:
    item_idx = item_to_idx[anime_id]
    scores = cosine_similarity(content_matrix[item_idx], content_matrix).ravel()
    scores[item_idx] = -np.inf

    top_indices = np.argpartition(scores, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    anime_ids = [idx_to_item[int(idx)] for idx in top_indices]

    result = anime[anime["anime_id"].isin(anime_ids)].copy()
    result["content_score"] = result["anime_id"].map(
        {idx_to_item[int(idx)]: float(scores[idx]) for idx in top_indices}
    )
    return result.sort_values("content_score", ascending=False)


def find_anime_id_by_name(anime: pd.DataFrame, anime_name: str) -> int:
    """Find the best anime_id match from a full or partial anime name."""
    normalized_name = anime_name.strip().casefold()
    names = anime["name"].fillna("").str.casefold()

    exact_matches = anime[names == normalized_name]
    if not exact_matches.empty:
        return int(exact_matches.iloc[0]["anime_id"])

    partial_matches = anime[names.str.contains(normalized_name, regex=False)]
    if partial_matches.empty:
        raise ValueError(f"No anime found with name containing: {anime_name}")

    best_match = partial_matches.sort_values(["rating", "members"], ascending=False).iloc[0]
    print(f"matched anime name: {best_match['name']} (anime_id={best_match['anime_id']})")
    return int(best_match["anime_id"])


def tune_bpr_with_optuna(data: PreparedData, trials: int = 10, k: int = 10) -> dict:
    import optuna

    train_items, test_items = train_test_split_by_user(data.user_positive_items)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "factors": trial.suggest_categorical("factors", [16, 32, 64]),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
            "regularization": trial.suggest_float("regularization", 1e-4, 1e-2, log=True),
            "epochs": trial.suggest_int("epochs", 3, 12),
        }

        model = BPRRecommender(
            n_users=len(data.user_to_idx),
            n_items=len(data.item_to_idx),
            **params,
        ).fit(train_items)

        return recall_at_k(model, train_items, test_items, k=k)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)
    print("best recall:", study.best_value)
    print("best params:", study.best_params)
    return study.best_params


def show_bpr_recommendations(data: PreparedData, model: BPRRecommender, user_id: int, top_k: int) -> pd.DataFrame:
    user_idx = data.user_to_idx[user_id]
    recommendations = model.recommend(user_idx, data.user_positive_items[user_idx], top_k=top_k)
    score_by_anime_id = {data.idx_to_item[item_idx]: score for item_idx, score in recommendations}

    result = data.anime[data.anime["anime_id"].isin(score_by_anime_id)].copy()
    result["bpr_score"] = result["anime_id"].map(score_by_anime_id)
    return result.sort_values("bpr_score", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-ratings", type=int, default=200_000)
    parser.add_argument("--min-rating", type=int, default=7)
    parser.add_argument("--trials", type=int, default=0)
    parser.add_argument("--user-id", type=int, default=None)
    parser.add_argument("--anime-id", type=int, default=5114)
    parser.add_argument("--anime-name", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    data = load_data(max_ratings=args.max_ratings, min_rating=args.min_rating)
    print(f"users={len(data.user_to_idx)} items={len(data.item_to_idx)}")

    params = {
        "factors": 32,
        "learning_rate": 0.03,
        "regularization": 0.002,
        "epochs": 5,
    }
    if args.trials > 0:
        params = tune_bpr_with_optuna(data, trials=args.trials, k=args.top_k)

    model = BPRRecommender(
        n_users=len(data.user_to_idx),
        n_items=len(data.item_to_idx),
        **params,
    ).fit(data.user_positive_items)

    if args.user_id is None:
        print("\nBPR model trained. Add --user-id <id> to print collaborative recommendations.")
    elif args.user_id not in data.user_to_idx:
        print(
            f"\nBPR model trained, but user_id={args.user_id} has no positive ratings "
            f"in this sampled data. Try increasing --max-ratings or using another user_id."
        )
    else:
        print("\nBPR recommendations")
        print(show_bpr_recommendations(data, model, args.user_id, args.top_k)[["name", "genre", "bpr_score"]])

    content_matrix = build_content_matrix(data.anime, data.item_to_idx)
    anime_id = args.anime_id
    if args.anime_name is not None:
        anime_id = find_anime_id_by_name(data.anime, args.anime_name)

    if anime_id in data.item_to_idx:
        print("\nContent-based similar anime")
        print(
            similar_anime_by_content(
                anime_id,
                data.anime,
                data.item_to_idx,
                data.idx_to_item,
                content_matrix,
                args.top_k,
            )[["name", "genre", "content_score"]]
        )


if __name__ == "__main__":
    main()
