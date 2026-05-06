import streamlit as st
import joblib
import pandas as pd

# --- 1. TẢI DỮ LIỆU TỪ FILE PKL ---
@st.cache_data # Lệnh này giúp web chỉ load dữ liệu 1 lần, chạy siêu nhanh
def load_data():
    matrix = joblib.load('hybrid_model.pkl')
    a_dict = joblib.load('anime_dict.pkl')
    return matrix, a_dict

similarity_matrix, anime_dict = load_data()

# Tạo một từ điển ngược (Tên phim -> ID) để lúc tìm kiếm cho dễ
name_to_id = {name: anime_id for anime_id, name in anime_dict.items()}
# Lấy danh sách toàn bộ tên phim để đưa vào ô tìm kiếm
anime_names = list(name_to_id.keys())

# --- 2. THIẾT KẾ GIAO DIỆN WEB ---
st.title("🍿 Hệ Thống Gợi Ý Anime Tối Ưu (Hybrid)")
st.write("Được cung cấp sức mạnh bởi Thuật toán Content-Based & Collaborative Filtering")

# Tạo ô tìm kiếm (Selectbox có hỗ trợ gõ text)
selected_anime = st.selectbox("🔍 Nhập hoặc chọn một bộ Anime bạn thích:", anime_names)

# Nút bấm chạy thuật toán
if st.button("TÌM PHIM TƯƠNG TỰ"):
    # --- 3. LOGIC XỬ LÝ ---
    # Lấy ID của bộ phim người dùng vừa chọn
    anime_id = name_to_id[selected_anime]
    
    # Lấy ra cột điểm số của phim đó trong Ma trận
    
    scores = similarity_matrix[anime_id]
    
    # Sắp xếp điểm số từ cao xuống thấp và lấy Top 6 (Bỏ qua phim đầu tiên vì là chính nó)
    top_ids = scores.sort_values(ascending=False).index[1:6]
    
    # --- 4. HIỂN THỊ KẾT QUẢ ---
    st.success(f"🎬 Gợi ý Top 5 phim dành cho bạn (dựa trên '{selected_anime}'):")
    
    # In ra màn hình
    for i, rec_id in enumerate(top_ids, 1):
        rec_name = anime_dict[rec_id]
        match_score = scores[rec_id] * 100
        
        st.write(f"**{i}. {rec_name}**")
        # Vẽ một thanh tiến trình (progress bar) nhỏ để thể hiện % độ giống nhau
        st.progress(scores[rec_id], text=f"Độ phù hợp: {match_score:.1f}%")