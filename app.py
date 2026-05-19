from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import streamlit as st
import html
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from als_recommender import ALSConfig, ExplicitALSRecommender

import requests

#lấy đường dẫn tới các file
BASE_DIR = Path(__file__).resolve().parent
RATINGS_PATH = BASE_DIR / "anime data" / "rating.csv"
ANIME_PATH = BASE_DIR / "anime data" / "anime.csv"
MODEL_DIR = BASE_DIR / "artifacts" / "als_model"
MODEL_FILENAME = "als_model.pkl"


st.set_page_config(
    page_title="Anime Hybrid Recommender",
    layout="wide",
    initial_sidebar_state="collapsed",
)

#version
NAME_CLEANING_VERSION = "v2"
RECOMMENDER_VERSION = "strict_title_similarity_v3"


DEFAULT_TOP_K = 5
DEFAULT_CONTENT_WEIGHT = 0.4
DEFAULT_TITLE_WEIGHT = 0.3
POSTER_CACHE_VERSION = "v4_retry_no_failed_cache"

#chuẩn hóa lại tên anime do có các kí tự lỗi, thừa
def clean_anime_name(name: object) -> str:
    if pd.isna(name):
        return ""

    cleaned = html.unescape(str(name)).strip()
    cleaned = cleaned.strip('"\' ')
    cleaned = re.sub(r"^\.?quot;\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*[\"']?\.hack//", "hack//", cleaned, flags=re.IGNORECASE)
    return cleaned


def format_anime_label(anime_id: object, anime_name: object) -> str:
    name = clean_anime_name(anime_name) or str(anime_id)
    return f"{name} (ID: {int(anime_id)})"


@st.cache_data
#load data từ anime.csv và rating.csv và làm sạch
def load_data(
    max_rows: int | None = None,
    cleaning_version: str = NAME_CLEANING_VERSION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _ = cleaning_version
    ratings = pd.read_csv(RATINGS_PATH)
    anime = pd.read_csv(ANIME_PATH)
    anime["name"] = anime["name"].apply(clean_anime_name)
    #giới hạn số rows trong dataset rating
    if max_rows is not None and max_rows > 0:
        ratings = ratings.head(max_rows).copy()
    return ratings, anime

#lưu tham số để không phải train lại
@st.cache_resource
def train_model(
    max_rows: int | None,
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


@st.cache_resource
#load model có sẵn thay vì phải train
def load_saved_model(
    model_dir: str,
    model_filename: str = MODEL_FILENAME,
    recommender_version: str = RECOMMENDER_VERSION,
) -> tuple[ExplicitALSRecommender, pd.DataFrame]:
    
    _ = recommender_version
    model = ExplicitALSRecommender.load(model_dir, model_filename=model_filename)
    _, anime_df = load_data(max_rows=None, cleaning_version=NAME_CLEANING_VERSION)
    return model, anime_df

#chuẩn hóa dữ liệu anime
def prepare_anime_options(
    anime_df: pd.DataFrame,
    trained_item_ids: set[int] | None = None,
) -> pd.DataFrame:
    
    #loại bỏ dữ liệu rác
    anime_lookup = anime_df.copy()
    anime_lookup["anime_id"] = pd.to_numeric(anime_lookup["anime_id"], errors="coerce")
    anime_lookup = anime_lookup.dropna(subset=["anime_id", "name"])
    anime_lookup["anime_id"] = anime_lookup["anime_id"].astype(int)
    anime_lookup = anime_lookup.drop_duplicates(subset=["anime_id"])
    #giữ lại các anime đã được model học
    if trained_item_ids is not None:
        anime_lookup = anime_lookup[anime_lookup["anime_id"].isin(trained_item_ids)].copy()
    
    #sắp xếp theo name
    return anime_lookup.sort_values("name")


@st.cache_resource
#biến đổi thông tin dạng chữ thành ma trận số
def build_content_lookup_and_matrix(anime_df: pd.DataFrame) -> tuple[pd.DataFrame, csr_matrix]:
    anime_lookup = prepare_anime_options(anime_df).reset_index(drop=True)

    #xử lí phần genre bằng tfidf
    genre_text = anime_lookup.get("genre", pd.Series("", index=anime_lookup.index))
    genre_text = genre_text.fillna("").astype(str).str.replace(",", " ", regex=False)
    if genre_text.str.strip().any():
        genre_features = TfidfVectorizer(min_df=1).fit_transform(genre_text)
    else:
        genre_features = csr_matrix((len(anime_lookup), 0), dtype=np.float32)

    #xử lí phần type_series bằng one-hot encoding
    type_series = anime_lookup.get("type", pd.Series("Unknown", index=anime_lookup.index))
    type_dummies = pd.get_dummies(type_series.fillna("Unknown").astype(str), dtype=np.float32)
    type_features = csr_matrix(type_dummies.to_numpy(dtype=np.float32))

    #xử lí phần rating, members bằng min-max normalization
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

    #gộp các phần thành ma trận và chuẩn hóa
    content_matrix = normalize(hstack([genre_features, type_features, numeric_features]).tocsr())
    return anime_lookup, content_matrix

#chuẩn hóa tiêu đề về 1 dạng thống nhất
def normalize_title(title: object) -> str:
    if pd.isna(title):
        return ""

    text = html.unescape(str(title)).casefold()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

#tính toán tỉ lệ trùng nhau ở tiêu đề anime
def shared_prefix_ratio(query_title: str, candidate_title: str) -> float:
    #tách từ
    query_tokens = query_title.split()
    candidate_tokens = candidate_title.split()
    if not query_tokens or not candidate_tokens:
        return 0.0

    #đếm số từ trùng nhau
    prefix_len = 0
    for query_token, candidate_token in zip(query_tokens, candidate_tokens):
        if query_token != candidate_token:
            break
        prefix_len += 1

    #nếu bộ phim có từ 2 từ trở lên thì phần trùng nhau phải có 2 từ trở lên
    min_len = min(len(query_tokens), len(candidate_tokens))
    if prefix_len < min(3, min_len):
        return 0.0

    #trả về tỉ lệ dựa trên tên dài hơn
    return prefix_len / max(len(query_tokens), len(candidate_tokens))

#so sánh tiêu đề của anime này với toàn bộ anime khác
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
        shared_prefix_score = shared_prefix_ratio(query_title, candidate_title)
        shorter_token_count = min(len(query_title.split()), len(candidate_title.split()))

        if candidate_title == query_title:
            score = 1.0
        elif candidate_title.startswith(f"{query_title} ") or query_title.startswith(f"{candidate_title} "):
            score = 0.95
        elif shared_prefix_score > 0:
            score = 0.85 + 0.1 * shared_prefix_score
        elif shorter_token_count >= 2 and (
            f" {query_title} " in f" {candidate_title} "
            or f" {candidate_title} " in f" {query_title} "
        ):
            score = 0.9
        else:
            token_score = (
                len(query_tokens & candidate_tokens) / len(query_tokens | candidate_tokens)
                if query_tokens and candidate_tokens
                else 0.0
            )
            score = token_score if token_score >= 0.5 else 0.0

        scores[idx] = float(score)

    scores[query_idx] = -np.inf
    return scores


def recompute_title_similarity(
    similar_df: pd.DataFrame,
    anime_lookup: pd.DataFrame,
    query_anime_id: int,
) -> pd.DataFrame:
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


def rerank_with_current_title_rules(
    similar_df: pd.DataFrame,
    anime_lookup: pd.DataFrame,
    query_anime_id: int,
    top_k: int,
    content_weight: float,
    title_weight: float,
) -> pd.DataFrame:
    similar_df = recompute_title_similarity(similar_df, anime_lookup, query_anime_id)

    has_hybrid_columns = {"als_similarity", "content_similarity", "title_similarity"}.issubset(similar_df.columns)
    if has_hybrid_columns:
        if "hybrid_score" in similar_df.columns:
            similar_df["hybrid_score"] = similar_df["hybrid_score"].astype("float64")

        title_weight = float(np.clip(title_weight, 0.0, 1.0))
        content_share = float(np.clip(content_weight, 0.0, 1.0))
        base_weight = 1.0 - title_weight
        als_weight = base_weight * (1.0 - content_share)
        effective_content_weight = base_weight * content_share

        trained_mask = similar_df["als_similarity"].notna() & similar_df["content_similarity"].notna()
        updated_scores = (
            als_weight * similar_df.loc[trained_mask, "als_similarity"].astype("float64").to_numpy()
            + effective_content_weight * similar_df.loc[trained_mask, "content_similarity"].astype("float64").to_numpy()
            + title_weight * similar_df.loc[trained_mask, "title_similarity"].fillna(0.0).astype("float64").to_numpy()
        )
        similar_df.loc[trained_mask, "hybrid_score"] = updated_scores

        metadata_mask = similar_df["als_similarity"].isna() & similar_df["content_similarity"].isna()
        similar_df = similar_df[~metadata_mask | (similar_df["title_similarity"].fillna(0.0) >= 0.85)].copy()

    if "hybrid_score" in similar_df.columns:
        similar_df = similar_df.sort_values("hybrid_score", ascending=False)

    return similar_df.head(top_k).reset_index(drop=True)


def existing_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]

#tính độ tương đồng
def content_similar_anime(
    anime_lookup: pd.DataFrame,
    content_matrix: csr_matrix,
    query_anime_id: int,
    top_k: int,
    title_weight: float = 0.3,
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

    #lấy top k phần tử điểm cao nhất
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

#phân chia thể loại dựa theo tâm trạng người dùng muốn xem bộ anime như nào
MOOD_GENRE_MAP = {
    "Không chọn": [],
    "Nhẹ nhàng": ["Comedy", "Slice of Life", "Romance"],
    "Hành động": ["Action", "Adventure", "Super Power", "Martial Arts"],
    "Drama": ["Drama", "Romance", "School"],
    "Hồi hộp / bí ẩn": ["Mystery", "Psychological", "Thriller", "Horror"],
    "Phiêu lưu / fantasy": ["Adventure", "Fantasy", "Magic", "Supernatural"],
}

#quét toàn bộ dữ liệu để lấy tất cả thể loại anime
def get_available_genres(anime_lookup: pd.DataFrame) -> list[str]:
    genre_values = anime_lookup.get("genre", pd.Series(dtype=str)).dropna().astype(str)
    genres = {genre.strip() for value in genre_values for genre in value.split(",") if genre.strip()}
    return sorted(genres)

#tính mức độ phổ biến cho 1 anime
def add_popularity_features(
    anime_lookup: pd.DataFrame,
    selected_genres: list[str] | None = None,
) -> pd.DataFrame:
    
    #điểm dựa trên mức độ khớp thể loại
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

    #điểm dựa trên rating và lượng người xem
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

#lọc dữ liệu thành 1 tập các anime dựa theo đầu vào của người dùng
def filter_preference_segment(
    anime_lookup: pd.DataFrame,
    selected_genres: list[str],
    selected_types: list[str],
    mood: str,
) -> tuple[pd.DataFrame, list[str]]:
    mood_genres = MOOD_GENRE_MAP.get(mood, [])
    target_genres = list(dict.fromkeys([*selected_genres, *mood_genres]))
    segment = anime_lookup.copy()

    #lọc theo định dạng phim (Movie, TV, ...)
    if selected_types:
        type_text = segment.get("type", pd.Series("", index=segment.index)).fillna("").astype(str)
        segment = segment[type_text.isin(selected_types)].copy()

    #lọc theo thể loại
    if target_genres:
        genre_set = set(target_genres)
        genre_text = segment.get("genre", pd.Series("", index=segment.index)).fillna("").astype(str)
        segment = segment[
            genre_text.apply(lambda value: bool({part.strip() for part in value.split(",")} & genre_set))
        ].copy()

    #tính điểm rồi sắp xếp
    segment = add_popularity_features(segment, target_genres)
    segment = segment.sort_values(["genre_match_count", "popularity_score"], ascending=False)
    return segment.reset_index(drop=True), target_genres

#tính tỉ lệ alpha để cân bằng giữa sở thích cá nhân (content_based) và xu hướng chung
#tick ít -> alpha cao ->  dựa theo xu hướng chung nhiều hơn
def dynamic_content_alpha(n_known: int, c: float = 8.0, alpha_min: float = 0.2, alpha_max: float = 0.9) -> float:
    return float(np.clip(c / (c + max(n_known, 0)), alpha_min, alpha_max))

#tạo danh sách 10 bộ phim phù hợp nhất
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
    #lọc để lấy các anime phù hợp định dạng, thể loại, tâm trạng
    segment, target_genres = filter_preference_segment(
        anime_lookup=anime_lookup,
        selected_genres=selected_genres,
        selected_types=selected_types,
        mood=mood,
    )
    alpha = dynamic_content_alpha(len(known_anime_ids))
    if segment.empty:
        return pd.DataFrame(), alpha, "empty"

    #loại các anime đã xem
    id_to_idx = {anime_id: idx for idx, anime_id in enumerate(anime_lookup["anime_id"].tolist())}
    known_anime_ids = [int(anime_id) for anime_id in known_anime_ids if int(anime_id) in id_to_idx]
    candidate = segment[~segment["anime_id"].isin(known_anime_ids)].copy()

    #người dùng có chọn tick các anime => dựa trên sở thích để tìm phim
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
    #người dùng không tick anime nào => chỉ dựa trên độ phổ biến của anime
    else:
        candidate["content_score"] = np.nan
        candidate["recommendation_score"] = candidate["popularity_score"]
        reason = "segment_popular"

    #phân trang
    candidate = candidate.sort_values(
        ["recommendation_score", "genre_match_count", "popularity_score"],
        ascending=False,
    )
    start = max(int(refresh_page), 0) * top_k
    if start >= len(candidate):
        start = 0

    #lấy số lượng phim theo yêu cầu và lấy các cột thông tin cần thiết
    columns = ["anime_id", "name", "genre", "type", "rating", "members", "recommendation_score"]
    result = candidate.iloc[start : start + top_k][columns].copy()
    result["matched_genres"] = ", ".join(target_genres) if target_genres else "All"
    return result.reset_index(drop=True), alpha, reason

def _extract_jikan_image_url(payload: dict) -> str | None:
    images = payload.get("images", {})
    jpg_images = images.get("jpg", {})
    return jpg_images.get("image_url") or jpg_images.get("large_image_url") or jpg_images.get("small_image_url")


def _jikan_get(url: str, **kwargs) -> requests.Response:
    session = requests.Session()
    session.trust_env = False
    return session.get(url, **kwargs)


def _download_image_bytes(image_url: str | None) -> bytes | None:
    if not image_url:
        return None

    response = _jikan_get(image_url, timeout=8)
    if response.status_code != 200:
        return None
    if "image" not in response.headers.get("content-type", ""):
        return None
    return response.content


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def get_poster_bytes(anime_id: int, anime_name: str, cache_version: str = POSTER_CACHE_VERSION) -> bytes | None:
    _ = cache_version
    headers = {"User-Agent": "Anime-Recommender-System/1.0"}

    for attempt in range(3):
        try:
            response = _jikan_get(f"https://api.jikan.moe/v4/anime/{int(anime_id)}", headers=headers, timeout=8)
            if response.status_code == 200:
                data = response.json()
                image_bytes = _download_image_bytes(_extract_jikan_image_url(data.get("data", {})))
                if image_bytes:
                    return image_bytes

            response = _jikan_get(
                "https://api.jikan.moe/v4/anime",
                params={"q": anime_name, "limit": 1},
                headers=headers,
                timeout=8,
            )
            if response.status_code == 200:
                data = response.json()
                matches = data.get("data", [])
                if matches:
                    image_bytes = _download_image_bytes(_extract_jikan_image_url(matches[0]))
                    if image_bytes:
                        return image_bytes
        except requests.RequestException:
            pass

        time.sleep(0.5 * (attempt + 1))

    raise RuntimeError(f"Poster not available for anime_id={anime_id}")


def get_poster_batch(batch: pd.DataFrame) -> dict[int, bytes | None]:
    poster_map: dict[int, bytes | None] = {}
    rows = [(int(row["anime_id"]), str(row["name"])) for _, row in batch.iterrows()]
    if not rows:
        return poster_map

    max_workers = min(3, len(rows))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(get_poster_bytes, anime_id, anime_name): anime_id
            for anime_id, anime_name in rows
        }
        for future in as_completed(futures):
            anime_id = futures[future]
            try:
                poster_map[anime_id] = future.result()
            except Exception:
                poster_map[anime_id] = None

    return poster_map
    
#hiển thị bảng kết quả dưới dạng ô lưới các hình ảnh
def display_anime_cards(df, columns_per_row=5):
    """Hiển thị lưới anime với ảnh lấy theo tên"""
    n_results = len(df)
    if n_results == 0:
        return

    #chia mỗi hàng có columns_per_row bộ anime
    for i in range(0, n_results, columns_per_row):
        cols = st.columns(columns_per_row)
        batch = df.iloc[i : i + columns_per_row]
        poster_map = get_poster_batch(batch)
        
        for idx, (index, row) in enumerate(batch.iterrows()):
            with cols[idx]:
                poster_bytes = poster_map.get(int(row["anime_id"]))
                if poster_bytes:
                    st.image(poster_bytes, use_container_width=True)
                else:
                    st.markdown(
                        """
                        <div style="
                            width: 100%;
                            aspect-ratio: 225 / 320;
                            border: 1px solid rgba(255,255,255,0.18);
                            border-radius: 6px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            color: rgba(255,255,255,0.55);
                            background: rgba(255,255,255,0.04);
                            font-size: 0.9rem;
                        ">No poster</div>
                        """,
                        unsafe_allow_html=True,
                    )
                
                # Hiển thị tên anime rút gọn
                short_name = row["name"][:40] + "..." if len(row["name"]) > 40 else row["name"]
                display_name = format_anime_label(row["anime_id"], short_name)
                st.markdown(f"**{display_name}**")
                
                # Hiển thị điểm số nếu có[cite: 1]
                if 'hybrid_score' in row and pd.notna(row['hybrid_score']):
                    st.caption(f"Score: {row['hybrid_score']:.2f}")
                elif 'content_similarity' in row and pd.notna(row['content_similarity']):
                    st.caption(f"Content Sim: {row['content_similarity']:.2f}")
                elif 'recommendation_score' in row and pd.notna(row['recommendation_score']):
                    st.caption(f"Rec Score: {row['recommendation_score']:.2f}")

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

        #hiển thị poster
        display_anime_cards(similar_df, columns_per_row=5)
        return

    #Anime này đã có trong ALS model => dùng hybrid
    similar_df = model.hybrid_similar_anime(
        query_anime_id,
        anime_lookup,
        top_k=top_k + 10,
        content_weight=content_weight,
        title_weight=DEFAULT_TITLE_WEIGHT,
    )
    similar_df = rerank_with_current_title_rules(
        similar_df,
        anime_lookup,
        query_anime_id,
        top_k,
        content_weight,
        DEFAULT_TITLE_WEIGHT,
    )
    st.success(f"Top {top_k} anime similar to: {query_name}")

    #hiển thị poster
    display_anime_cards(similar_df, columns_per_row=5)
    

#gợi ý cho người mới
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
    display_anime_cards(recommendations_df, columns_per_row=5)
    
    return

#tiêu đề web
st.title("Anime Recommender System")
#mô tả hệ thống
st.write("Hybrid = ALS collaborative similarity + content similarity from genre, type, rating, and members.")
st.write("Power by Save AI")
st.markdown(
    """
    <style>
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

saved_model_exists = (MODEL_DIR / MODEL_FILENAME).exists()
top_k = DEFAULT_TOP_K
content_weight = DEFAULT_CONTENT_WEIGHT

if saved_model_exists:
    with st.spinner("Loading saved ALS model..."):
        model, anime_df = load_saved_model(str(MODEL_DIR), recommender_version=RECOMMENDER_VERSION)
else:
    st.error(f"Saved model not found: {MODEL_DIR / MODEL_FILENAME}")
    st.stop()

trained_item_ids = set(model.item_ids.tolist()) if model.item_ids is not None else set()
anime_lookup_full, content_matrix = build_content_lookup_and_matrix(anime_df)
anime_lookup = prepare_anime_options(anime_df, trained_item_ids=trained_item_ids)
tab_search, tab_id, tab_recommendations = st.tabs(["Search by name", "Search by anime_id", "User Recommendations"])

with tab_search:
    anime_name_map = dict(zip(anime_lookup_full["anime_id"], anime_lookup_full["name"]))
    selected_anime_id = st.selectbox(
        "Choose anime",
        anime_lookup_full["anime_id"].tolist(),
        format_func=lambda anime_id: format_anime_label(anime_id, anime_name_map.get(anime_id, anime_id)),
    )
    if st.button("Find similar anime", key="search_by_name"):
        query_anime_id = int(selected_anime_id)
        query_name = format_anime_label(query_anime_id, anime_name_map.get(query_anime_id, query_anime_id))
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
            query_name = (
                format_anime_label(query_anime_id, query_row.iloc[0]["name"])
                if not query_row.empty
                else str(query_anime_id)
            )
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
        format_func=lambda anime_id: format_anime_label(anime_id, starter_name_map.get(anime_id, anime_id)),
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
        format_func=lambda anime_id: format_anime_label(anime_id, seed_anime_name_map.get(anime_id, anime_id)),
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





