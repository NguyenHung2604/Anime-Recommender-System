from __future__ import annotations

from pathlib import Path
import re
from difflib import SequenceMatcher

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
RECOMMENDER_VERSION = "title_similarity_v1"


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
    recommender_version: str = RECOMMENDER_VERSION,
) -> tuple[ExplicitALSRecommender, pd.DataFrame]:
    _ = recommender_version
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


def normalize_title(title: object) -> str:
    if pd.isna(title):
        return ""

    text = html.unescape(str(title)).casefold()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_similarity_scores(anime_lookup: pd.DataFrame, query_idx: int) -> np.ndarray:
    query_title = normalize_title(anime_lookup.iloc[query_idx].get("name", ""))
    query_tokens = set(query_title.split())
    scores = np.zeros(len(anime_lookup), dtype=np.float32)

    if not query_title:
        return scores

    for idx, candidate_name in enumerate(anime_lookup.get("name", pd.Series("", index=anime_lookup.index))):
        candidate_title = normalize_title(candidate_name)
        if not candidate_title:
            continue

        candidate_tokens = set(candidate_title.split())
        token_score = (
            len(query_tokens & candidate_tokens) / len(query_tokens | candidate_tokens)
            if query_tokens and candidate_tokens
            else 0.0
        )
        sequence_score = SequenceMatcher(None, query_title, candidate_title).ratio()

        if candidate_title == query_title:
            score = 1.0
        elif candidate_title.startswith(f"{query_title} ") or query_title.startswith(f"{candidate_title} "):
            score = 0.95
        elif query_title in candidate_title or candidate_title in query_title:
            score = 0.9
        else:
            score = max(token_score, sequence_score * 0.75)

        scores[idx] = float(score)

    scores[query_idx] = -np.inf
    return scores


def add_title_similarity_if_missing(
    similar_df: pd.DataFrame,
    anime_lookup: pd.DataFrame,
    query_anime_id: int,
) -> pd.DataFrame:
    if "title_similarity" in similar_df.columns:
        return similar_df

    id_to_idx = {anime_id: idx for idx, anime_id in enumerate(anime_lookup["anime_id"].tolist())}
    if query_anime_id not in id_to_idx:
        similar_df = similar_df.copy()
        similar_df["title_similarity"] = np.nan
        return similar_df

    scores = title_similarity_scores(anime_lookup, id_to_idx[query_anime_id])
    score_by_id = {
        int(anime_id): float(scores[idx])
        for anime_id, idx in id_to_idx.items()
    }
    similar_df = similar_df.copy()
    similar_df["title_similarity"] = similar_df["anime_id"].map(score_by_id)
    return similar_df


def existing_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def content_similar_anime(
    anime_lookup: pd.DataFrame,
    content_matrix: csr_matrix,
    query_anime_id: int,
    top_k: int,
    title_weight: float = 0.55,
) -> pd.DataFrame:
    id_to_idx = {anime_id: idx for idx, anime_id in enumerate(anime_lookup["anime_id"].tolist())}
    if query_anime_id not in id_to_idx:
        raise ValueError(f"Anime id {query_anime_id} not found in anime metadata.")

    query_idx = id_to_idx[query_anime_id]
    content_scores = (content_matrix @ content_matrix[query_idx].T).toarray().ravel().astype(np.float32)
    title_scores = title_similarity_scores(anime_lookup, query_idx)
    title_weight = float(np.clip(title_weight, 0.0, 1.0))
    scores = (1.0 - title_weight) * content_scores + title_weight * title_scores
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
    result["metadata_similarity"] = content_scores[top_indices]
    result["title_similarity"] = title_scores[top_indices]
    return result.reset_index(drop=True)


MOOD_GENRE_MAP = {
    "Không chọn": [],
    "Nhẹ nhàng": ["Comedy", "Slice of Life", "Romance"],
    "Hanh động": ["Action", "Adventure", "Super Power", "Martial Arts"],
    "Drama": ["Drama", "Romance", "School"],
    "Hồi hộp / bí ẩn": ["Mystery", "Psychological", "Thriller", "Horror"],
    "Phiêu lưu / fantasy": ["Adventure", "Fantasy", "Magic", "Supernatural"],
}


def get_available_genres(anime_lookup: pd.DataFrame) -> list[str]:
    genre_values = anime_lookup.get("genre", pd.Series(dtype=str)).dropna().astype(str)
    genres = {genre.strip() for value in genre_values for genre in value.split(",") if genre.strip()}
    return sorted(genres)


def add_popularity_features(
    anime_lookup: pd.DataFrame,
    selected_genres: list[str] | None = None,
) -> pd.DataFrame:
    result = anime_lookup.copy()
    selected_genres = selected_genres or []
    selected_genre_set = set(selected_genres)

    genre_text = result.get("genre", pd.Series("", index=result.index)).fillna("").astype(str)
    if selected_genre_set:
        result["genre_match_count"] = genre_text.apply(
            lambda value: len({part.strip() for part in value.split(",")} & selected_genre_set)
        )
    else:
        result["genre_match_count"] = 0

    rating = pd.to_numeric(result.get("rating", pd.Series(0, index=result.index)), errors="coerce").fillna(0.0)
    members = pd.to_numeric(result.get("members", pd.Series(0, index=result.index)), errors="coerce").fillna(0.0)
    members = np.log1p(members)

    rating_range = rating.max() - rating.min()
    members_range = members.max() - members.min()
    rating_norm = (rating - rating.min()) / (rating_range if rating_range else 1.0)
    members_norm = (members - members.min()) / (members_range if members_range else 1.0)

    result["popularity_score"] = (
        0.62 * rating_norm
        + 0.33 * members_norm
        + 0.05 * np.minimum(result["genre_match_count"], 3) / 3
    )
    return result


def filter_preference_segment(
    anime_lookup: pd.DataFrame,
    selected_genres: list[str],
    selected_types: list[str],
    mood: str,
) -> tuple[pd.DataFrame, list[str]]:
    mood_genres = MOOD_GENRE_MAP.get(mood, [])
    target_genres = list(dict.fromkeys([*selected_genres, *mood_genres]))
    segment = anime_lookup.copy()

    if selected_types:
        type_text = segment.get("type", pd.Series("", index=segment.index)).fillna("").astype(str)
        segment = segment[type_text.isin(selected_types)].copy()

    if target_genres:
        genre_set = set(target_genres)
        genre_text = segment.get("genre", pd.Series("", index=segment.index)).fillna("").astype(str)
        segment = segment[
            genre_text.apply(lambda value: bool({part.strip() for part in value.split(",")} & genre_set))
        ].copy()

    segment = add_popularity_features(segment, target_genres)
    segment = segment.sort_values(["genre_match_count", "popularity_score"], ascending=False)
    return segment.reset_index(drop=True), target_genres


def dynamic_content_alpha(n_known: int, c: float = 8.0, alpha_min: float = 0.2, alpha_max: float = 0.9) -> float:
    return float(np.clip(c / (c + max(n_known, 0)), alpha_min, alpha_max))


def recommend_for_new_viewer(
    anime_lookup: pd.DataFrame,
    content_matrix: csr_matrix,
    selected_genres: list[str],
    selected_types: list[str],
    mood: str,
    known_anime_ids: list[int],
    top_k: int = 10,
    refresh_page: int = 0,
) -> tuple[pd.DataFrame, float, str]:
    segment, target_genres = filter_preference_segment(
        anime_lookup=anime_lookup,
        selected_genres=selected_genres,
        selected_types=selected_types,
        mood=mood,
    )
    alpha = dynamic_content_alpha(len(known_anime_ids))
    if segment.empty:
        return pd.DataFrame(), alpha, "empty"

    id_to_idx = {anime_id: idx for idx, anime_id in enumerate(anime_lookup["anime_id"].tolist())}
    known_anime_ids = [int(anime_id) for anime_id in known_anime_ids if int(anime_id) in id_to_idx]
    candidate = segment[~segment["anime_id"].isin(known_anime_ids)].copy()

    if known_anime_ids:
        known_indices = [id_to_idx[anime_id] for anime_id in known_anime_ids]
        profile_vector = content_matrix[known_indices].mean(axis=0)
        content_scores = np.asarray(content_matrix @ profile_vector.T).ravel().astype(np.float32)
        candidate["content_score"] = candidate["anime_id"].map(
            {anime_id: float(content_scores[idx]) for anime_id, idx in id_to_idx.items()}
        )
        alpha = dynamic_content_alpha(len(known_anime_ids))
        candidate["recommendation_score"] = (
            alpha * candidate["content_score"].fillna(0.0)
            + (1.0 - alpha) * candidate["popularity_score"].fillna(0.0)
        )
        reason = "profile"
    else:
        candidate["content_score"] = np.nan
        candidate["recommendation_score"] = candidate["popularity_score"]
        reason = "segment_popular"

    candidate = candidate.sort_values(
        ["recommendation_score", "genre_match_count", "popularity_score"],
        ascending=False,
    )
    start = max(int(refresh_page), 0) * top_k
    if start >= len(candidate):
        start = 0

    columns = ["anime_id", "name", "genre", "type", "rating", "members", "recommendation_score"]
    result = candidate.iloc[start : start + top_k][columns].copy()
    result["matched_genres"] = ", ".join(target_genres) if target_genres else "All"
    return result.reset_index(drop=True), alpha, reason


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
            similar_df[
                existing_columns(
                    similar_df,
                    [
                        "anime_id",
                        "name",
                        "genre",
                        "type",
                        "content_similarity",
                        "metadata_similarity",
                        "title_similarity",
                    ],
                )
            ],
            use_container_width=True,
        )
        return

    similar_df = model.hybrid_similar_anime(
        query_anime_id,
        anime_lookup,
        top_k=top_k,
        content_weight=content_weight,
    )
    similar_df = add_title_similarity_if_missing(similar_df, anime_lookup, query_anime_id)
    st.success(f"Top {top_k} anime similar to: {query_name}")
    st.dataframe(
        similar_df[
            existing_columns(
                similar_df,
                [
                    "anime_id",
                    "name",
                    "genre",
                    "type",
                    "hybrid_score",
                    "als_similarity",
                    "content_similarity",
                    "title_similarity",
                ],
            )
        ],
        use_container_width=True,
    )


def show_new_viewer_recommendations(
    anime_lookup: pd.DataFrame,
    content_matrix: csr_matrix,
    selected_genres: list[str],
    selected_types: list[str],
    mood: str,
    known_anime_ids: list[int],
    refresh_page: int,
) -> None:
    recommendations_df, alpha, reason = recommend_for_new_viewer(
        anime_lookup=anime_lookup,
        content_matrix=content_matrix,
        selected_genres=selected_genres,
        selected_types=selected_types,
        mood=mood,
        known_anime_ids=known_anime_ids,
        top_k=10,
        refresh_page=refresh_page,
    )

    if recommendations_df.empty:
        st.warning("Không tìm thấy anime phù hợp với bộ lọc này. Hãy chọn ít thể loại/type hơn.")
        return

    if reason == "profile":
        st.info(f"Đã tạo hồ sơ từ {len(known_anime_ids)} anime bạn tick. Content alpha = {alpha:.2f}.")
    else:
        st.info("Bạn chưa tick anime quen thuộc, nên he thống dùng Top Popular trong các phân khúc bạn chọn.")

    st.success("10 anime gợi ý cho bạn")
    st.dataframe(
        recommendations_df[
            [
                "anime_id",
                "name",
                "genre",
                "type",
                "rating",
                "members",
                "recommendation_score",
            ]
        ],
        use_container_width=True,
    )
    return

st.title("Anime Recommender System")
st.write("Hybrid = ALS collaborative similarity + content similarity from genre, type, rating, and members.")
st.write("Power by Save AI")

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
    model, anime_df = train_model(max_rows, n_factors, n_iters, reg, int(random_state), RECOMMENDER_VERSION)

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
    st.subheader("Chào bạn, bạn thích xem gì hôm nay?")
    st.write("Chọn nhanh so thich, hệ thống se gợi ý 10 anime phù hợp.")

    available_genres = get_available_genres(anime_lookup_full)
    available_types = sorted(anime_lookup_full["type"].dropna().astype(str).unique().tolist())

    selected_genres = st.multiselect(
        "Thể loại bạn thích (chọn 2-3 mục)",
        available_genres,
        default=[genre for genre in ["Action", "Comedy", "Drama"] if genre in available_genres][:2],
    )
    selected_types = st.multiselect(
        "Định dạng anime bạn thích",
        available_types,
        default=[anime_type for anime_type in ["TV", "Movie"] if anime_type in available_types],
    )
    mood = st.selectbox("Mood / mục tiêu xem", list(MOOD_GENRE_MAP.keys()), index=0)

    starter_segment, _ = filter_preference_segment(
        anime_lookup_full,
        selected_genres=selected_genres,
        selected_types=selected_types,
        mood=mood,
    )
    starter_options = starter_segment.head(30)
    starter_name_map = dict(zip(starter_options["anime_id"], starter_options["name"]))
    known_anime_ids = st.multiselect(
        "Tick vài anime bạn đã biết (có thể bỏ qua)",
        starter_options["anime_id"].tolist(),
        format_func=lambda anime_id: starter_name_map.get(anime_id, str(anime_id)),
        help="Nếu tick 3-5 anime, hệ thống sẽ tạo content profile ban đầu. Nếu bỏ qua, sẽ dùng Top Popular theo phân khúc.",
    )

    refresh_page = st.number_input(
        "Làm mới / khám phá trang",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
        help="Tăng sô này để xem 10 gợi ý tiếp theo trong cung phân khúc.",
    )

    if st.button("Recommend 10 anime", key="get_new_viewer_recommendations"):
        try:
            show_new_viewer_recommendations(
                anime_lookup_full,
                content_matrix,
                selected_genres,
                selected_types,
                mood,
                [int(anime_id) for anime_id in known_anime_ids],
                int(refresh_page),
            )
        except ValueError as exc:
            st.error(str(exc))

    st.caption("The model is trained from rating.csv and anime metadata in anime.csv.")
    st.stop()

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
