from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st
import html
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from als_recommender import ALSConfig, ExplicitALSRecommender


BASE_DIR = Path(__file__).resolve().parent
RATINGS_PATH = BASE_DIR / "anime data" / "rating.csv"
ANIME_PATH = BASE_DIR / "anime data" / "anime.csv"


st.set_page_config(page_title="Anime Hybrid Recommender", layout="wide")

NAME_CLEANING_VERSION = "v2"


def clean_anime_name(name: object) -> str:
    if pd.isna(name):
        return ""

    cleaned = html.unescape(str(name)).strip()
    cleaned = cleaned.strip('"\' ')
    cleaned = re.sub(r"^\.?quot;\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*[\"']?\.hack//", "hack//", cleaned, flags=re.IGNORECASE)
    return cleaned


@st.cache_data
def load_data(
    max_rows: int | None = None,
    cleaning_version: str = NAME_CLEANING_VERSION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _ = cleaning_version
    ratings = pd.read_csv(RATINGS_PATH)
    anime = pd.read_csv(ANIME_PATH)
    anime["name"] = anime["name"].apply(clean_anime_name)
    if max_rows is not None and max_rows > 0:
        ratings = ratings.head(max_rows).copy()
    return ratings, anime


@st.cache_resource
def train_model(
    max_rows: int,
    n_factors: int,
    n_iters: int,
    reg: float,
    random_state: int,
) -> tuple[ExplicitALSRecommender, pd.DataFrame]:
    ratings_df, anime_df = load_data(max_rows=max_rows, cleaning_version=NAME_CLEANING_VERSION)
    model = ExplicitALSRecommender(
        ALSConfig(
            n_factors=n_factors,
            n_iters=n_iters,
            reg=reg,
            random_state=random_state,
        )
    )
    model.fit(ratings_df)
    return model, anime_df


def prepare_anime_options(
    anime_df: pd.DataFrame,
    trained_item_ids: set[int] | None = None,
) -> pd.DataFrame:
    anime_lookup = anime_df.copy()
    anime_lookup["anime_id"] = pd.to_numeric(anime_lookup["anime_id"], errors="coerce")
    anime_lookup = anime_lookup.dropna(subset=["anime_id", "name"])
    anime_lookup["anime_id"] = anime_lookup["anime_id"].astype(int)
    anime_lookup = anime_lookup.drop_duplicates(subset=["anime_id"])
    if trained_item_ids is not None:
        anime_lookup = anime_lookup[anime_lookup["anime_id"].isin(trained_item_ids)].copy()
    return anime_lookup.sort_values("name")


@st.cache_resource
def build_content_lookup_and_matrix(anime_df: pd.DataFrame) -> tuple[pd.DataFrame, csr_matrix]:
    anime_lookup = prepare_anime_options(anime_df).reset_index(drop=True)

    genre_text = anime_lookup.get("genre", pd.Series("", index=anime_lookup.index))
    genre_text = genre_text.fillna("").astype(str).str.replace(",", " ", regex=False)
    if genre_text.str.strip().any():
        genre_features = TfidfVectorizer(min_df=1).fit_transform(genre_text)
    else:
        genre_features = csr_matrix((len(anime_lookup), 0), dtype=np.float32)

    type_series = anime_lookup.get("type", pd.Series("Unknown", index=anime_lookup.index))
    type_dummies = pd.get_dummies(type_series.fillna("Unknown").astype(str), dtype=np.float32)
    type_features = csr_matrix(type_dummies.to_numpy(dtype=np.float32))

    numeric_columns = [column for column in ["rating", "members"] if column in anime_lookup.columns]
    if numeric_columns:
        numeric = anime_lookup[numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        numeric_values = numeric.to_numpy(dtype=np.float32)
        mins = numeric_values.min(axis=0)
        ranges = numeric_values.max(axis=0) - mins
        ranges[ranges == 0] = 1.0
        numeric_values = (numeric_values - mins) / ranges
        numeric_features = csr_matrix(numeric_values)
    else:
        numeric_features = csr_matrix((len(anime_lookup), 0), dtype=np.float32)

    content_matrix = normalize(hstack([genre_features, type_features, numeric_features]).tocsr())
    return anime_lookup, content_matrix


def content_similar_anime(
    anime_lookup: pd.DataFrame,
    content_matrix: csr_matrix,
    query_anime_id: int,
    top_k: int,
) -> pd.DataFrame:
    id_to_idx = {anime_id: idx for idx, anime_id in enumerate(anime_lookup["anime_id"].tolist())}
    if query_anime_id not in id_to_idx:
        raise ValueError(f"Anime id {query_anime_id} not found in anime metadata.")

    query_idx = id_to_idx[query_anime_id]
    scores = (content_matrix @ content_matrix[query_idx].T).toarray().ravel().astype(np.float32)
    scores[query_idx] = -np.inf

    n_items = len(scores)
    effective_top_k = min(max(int(top_k), 1), n_items - 1)
    if effective_top_k <= 0:
        return pd.DataFrame(columns=["anime_id", "name", "genre", "type", "content_similarity"])

    if effective_top_k >= n_items:
        top_indices = np.argsort(-scores)
    else:
        top_indices = np.argpartition(-scores, effective_top_k)[:effective_top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

    result = anime_lookup.iloc[top_indices][["anime_id", "name", "genre", "type"]].copy()
    result["content_similarity"] = scores[top_indices]
    return result.reset_index(drop=True)


def show_query_result(
    model: ExplicitALSRecommender,
    anime_lookup: pd.DataFrame,
    content_matrix: csr_matrix,
    query_anime_id: int,
    query_name: str,
    top_k: int,
    content_weight: float,
) -> None:
    if query_anime_id not in model.item_to_idx:
        st.info("Anime này chưa có trong ALS model. Đang chuyển sang content-based only.")
        similar_df = content_similar_anime(
            anime_lookup=anime_lookup,
            content_matrix=content_matrix,
            query_anime_id=query_anime_id,
            top_k=top_k,
        )
        st.success(f"Top {top_k} anime similar to: {query_name} (content-based)")
        st.dataframe(
            similar_df[["anime_id", "name", "genre", "type", "content_similarity"]],
            use_container_width=True,
        )
        return

    similar_df = model.hybrid_similar_anime(
        query_anime_id,
        anime_lookup,
        top_k=top_k,
        content_weight=content_weight,
    )
    st.success(f"Top {top_k} anime similar to: {query_name}")
    st.dataframe(
        similar_df[
            [
                "anime_id",
                "name",
                "genre",
                "type",
                "hybrid_score",
                "als_similarity",
                "content_similarity",
            ]
        ],
        use_container_width=True,
    )


def show_recommendations(
    model: ExplicitALSRecommender,
    anime_lookup: pd.DataFrame,
    content_matrix: csr_matrix,
    user_id: int,
    top_k: int,
    cold_start_anime_id: int,
) -> None:
    if user_id not in model.user_to_idx:
        seed_row = anime_lookup[anime_lookup["anime_id"] == cold_start_anime_id]
        seed_name = seed_row.iloc[0]["name"] if not seed_row.empty else str(cold_start_anime_id)
        st.info(
            "User này chưa có rating trong dữ liệu train. "
            f"Đang dùng content-based từ anime seed: {seed_name}."
        )
        recommendations_df = content_similar_anime(
            anime_lookup=anime_lookup,
            content_matrix=content_matrix,
            query_anime_id=cold_start_anime_id,
            top_k=top_k,
        )
        st.success(f"Top {top_k} content-based recommendations")
        st.dataframe(
            recommendations_df[["anime_id", "name", "genre", "type", "content_similarity"]],
            use_container_width=True,
        )
        return

    recommendations_df = model.recommend(user_id, anime_lookup, top_k=top_k)
    st.success(f"Top {top_k} recommendations for user {user_id}")
    st.dataframe(
        recommendations_df[
            [
                "anime_id",
                "name",
                "genre",
                "type",
                "predicted_rating",
            ]
        ],
        use_container_width=True,
    )


st.title("Anime Hybrid Recommender")
st.write("Hybrid = ALS collaborative similarity + content similarity from genre, type, rating, and members.")

with st.sidebar:
    st.header("Training")
    max_rows = st.slider("Rating rows", min_value=5000, max_value=100000, value=10000, step=5000)
    n_factors = st.slider("ALS latent factors", min_value=8, max_value=64, value=16, step=8)
    n_iters = st.slider("ALS iterations", min_value=2, max_value=10, value=4, step=1)
    reg = st.slider("Regularization", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)

    st.header("Hybrid")
    top_k = st.slider("Number of similar anime", min_value=5, max_value=20, value=5, step=1)
    content_weight = st.slider(
        "Content weight",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="0 = ALS only, 1 = content only. Try 0.5 to 0.7 for search-by-anime.",
    )


with st.spinner("Training ALS model..."):
    model, anime_df = train_model(max_rows, n_factors, n_iters, reg, int(random_state))

trained_item_ids = set(model.item_ids.tolist()) if model.item_ids is not None else set()
anime_lookup_full, content_matrix = build_content_lookup_and_matrix(anime_df)
anime_lookup = prepare_anime_options(anime_df, trained_item_ids=trained_item_ids)
tab_search, tab_id, tab_recommendations = st.tabs(["Search by name", "Search by anime_id", "User Recommendations"])

with tab_search:
    anime_name_map = dict(zip(anime_lookup_full["anime_id"], anime_lookup_full["name"]))
    selected_anime_id = st.selectbox(
        "Choose anime",
        anime_lookup_full["anime_id"].tolist(),
        format_func=lambda anime_id: anime_name_map.get(anime_id, str(anime_id)),
    )
    if st.button("Find similar anime", key="search_by_name"):
        query_anime_id = int(selected_anime_id)
        query_name = anime_name_map.get(query_anime_id, str(query_anime_id))
        try:
            show_query_result(
                model,
                anime_lookup_full,
                content_matrix,
                query_anime_id,
                query_name,
                top_k,
                content_weight,
            )
        except ValueError as exc:
            st.error(str(exc))

with tab_id:
    anime_id_input = st.number_input(
        "Enter anime_id",
        min_value=1,
        value=int(anime_lookup_full.iloc[0]["anime_id"]),
        step=1,
    )
    if st.button("Find similar anime", key="search_by_id"):
        try:
            query_anime_id = int(anime_id_input)
            query_row = anime_lookup_full[anime_lookup_full["anime_id"] == query_anime_id]
            query_name = query_row.iloc[0]["name"] if not query_row.empty else str(query_anime_id)
            show_query_result(
                model,
                anime_lookup_full,
                content_matrix,
                query_anime_id,
                query_name,
                top_k,
                content_weight,
            )
        except ValueError as exc:
            st.error(str(exc))

with tab_recommendations:
    seed_anime_name_map = dict(zip(anime_lookup_full["anime_id"], anime_lookup_full["name"]))
    user_id_input = st.number_input(
        "Enter user_id",
        min_value=1,
        value=1,
        step=1,
    )
    cold_start_anime_id = st.selectbox(
        "Seed anime for new users (content-based)",
        anime_lookup_full["anime_id"].tolist(),
        index=0,
        format_func=lambda anime_id: seed_anime_name_map.get(anime_id, str(anime_id)),
        help="Dùng khi user chưa có rating trong dữ liệu train.",
    )
    if st.button("Get recommendations", key="get_recommendations"):
        try:
            show_recommendations(
                model,
                anime_lookup_full,
                content_matrix,
                int(user_id_input),
                top_k,
                int(cold_start_anime_id),
            )
        except ValueError as exc:
            st.error(str(exc))

st.caption("The model is trained from rating.csv and anime metadata in anime.csv.")
