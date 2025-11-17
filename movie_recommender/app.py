import streamlit as st
import requests
import pandas as pd
import os

# ---------------------- НАСТРОЙКИ ----------------------
st.set_page_config(page_title="Рекомендательная система фильмов", layout="wide")
API_BASE_URL = "http://localhost:8000"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------- СТИЛИ ----------------------
def apply_modern_theme():
    st.markdown(
        """
        <style>
        :root{
          --bg:#0b1020;
          --card:rgba(255,255,255,0.06);
          --border:rgba(255,255,255,0.10);
          --text:#e5e7eb;
          --muted:#a1a1aa;
          --accent:#7c3aed;
          --accent2:#22d3ee;
        }
        [data-testid="stAppViewContainer"]{
          background:
            radial-gradient(900px 400px at 15% 5%, rgba(124,58,237,0.12), transparent),
            radial-gradient(1000px 500px at 90% 20%, rgba(34,211,238,0.10), transparent),
            var(--bg);
          color: var(--text);
        }
        [data-testid="stHeader"] { background: transparent; }
        .stButton>button{
          background: linear-gradient(135deg, rgba(124,58,237,0.35), rgba(34,211,238,0.35));
          color: white; border: 1px solid var(--border);
          border-radius: 10px; padding: 8px 12px; font-weight: 600;
        }
        .stButton>button:hover{ filter: brightness(1.15); }
        </style>
        """,
        unsafe_allow_html=True
    )

# ---------------------- API ФУНКЦИИ ----------------------
def check_api_connection():
    try:
        response = requests.get(f"{API_BASE_URL}/")
        return response.status_code == 200
    except:
        return False

def get_top_movies(n=10):
    try:
        r = requests.get(f"{API_BASE_URL}/api/recommendations/top-movies?n={n}")
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        st.error(f"Ошибка: {e}")
        return None

def get_recommendations_by_genre(genre, n=10):
    try:
        r = requests.get(f"{API_BASE_URL}/api/recommendations/genre/{genre}?n={n}")
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        st.error(f"Ошибка: {e}")
        return None

def get_content_recommendations(movie_title, n=10):
    try:
        r = requests.get(f"{API_BASE_URL}/api/recommendations/similar/{movie_title}?n={n}")
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        st.error(f"Ошибка: {e}")
        return None

def get_collaborative_recommendations(user_id, n=10):
    try:
        r = requests.get(f"{API_BASE_URL}/api/recommendations/collaborative/{user_id}?n={n}")
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        st.error(f"Ошибка: {e}")
        return None

def get_personal_recommendations(n=10):
    try:
        r = requests.get(f"{API_BASE_URL}/api/recommendations/personal?n={n}")
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        st.error(f"Ошибка: {e}")
        return None

def get_user_ratings(user_id):
    try:
        r = requests.get(f"{API_BASE_URL}/api/user/{user_id}/ratings")
        return r.json() if r.status_code == 200 else None
    except:
        return None

# ---------------------- ВИЗУАЛИЗАЦИЯ ----------------------
def display_movies(movies, show_predicted_rating=False):
    if not movies:
        st.error("Нет подходящих фильмов")
        return
    cols = st.columns(2)
    for i, m in enumerate(movies):
        with cols[i % 2]:
            st.write(f"**{m.get('title', '')}**")
            st.write(f"Жанры: {m.get('genres', '')}")
            rating = (
                m.get("predicted_rating") if show_predicted_rating else
                m.get("weighted_rating") or m.get("rating_mean") or m.get("rating")
            )
            if rating:
                st.metric("Рейтинг", f"{rating * 2:.1f}/10")
            st.divider()

def display_ratings_table(df):
    df["Оценка (/10)"] = (df["rating"] * 2).round(1)
    st.dataframe(df[["title", "genres", "Оценка (/10)"]], use_container_width=True)

# ---------------------- ПЕРСОНАЛЬНЫЕ ----------------------
def retrain_model():
    try:
        r = requests.post(f"{API_BASE_URL}/api/retrain")
        if r.status_code == 200:
            st.success("Модель переобучена.")
        else:
            st.warning("Не удалось переобучить модель.")
    except:
        st.warning("Ошибка при переобучении модели.")

def create_initial_ratings():
    st.subheader("Создание начальных оценок")
    top_movies = get_top_movies(20)
    if not top_movies:
        st.error("Не удалось загрузить фильмы.")
        return
    ratings = []
    for movie in top_movies[:10]:
        st.write(f"**{movie['title']}** — {movie['genres']}")
        rate = st.slider(f"Оценка ({movie['title']})", 0.0, 5.0, 0.0, 0.5)
        if rate > 0:
            ratings.append({
                "movieId": movie["movie_id"],
                "title": movie["title"],
                "genres": movie["genres"],
                "rating": rate
            })
    if st.button("Сохранить оценки") and len(ratings) >= 3:
        pd.DataFrame(ratings).to_csv(os.path.join(BASE_DIR, "personal_ratings_detailed.csv"), index=False)
        st.success("Оценки сохранены.")
        retrain_model()
        st.rerun()

def add_single_rating():
    st.subheader("Добавить оценку фильма")
    top_movies = get_top_movies(100)
    if not top_movies:
        return
    rated_ids = set()
    pr_path = os.path.join(BASE_DIR, "personal_ratings_detailed.csv")
    if os.path.exists(pr_path):
        rated_ids = set(pd.read_csv(pr_path)["movieId"].astype(int))
    filtered = [m for m in top_movies if int(m["movie_id"]) not in rated_ids]
    if not filtered:
        st.info("Все фильмы уже оценены.")
        return
    selected = st.selectbox("Выберите фильм", [m["title"] for m in filtered])
    rate = st.slider("Ваша оценка", 0.5, 5.0, 2.5, 0.5)
    if st.button("Сохранить"):
        df = pd.DataFrame([{
            "movieId": next(m["movie_id"] for m in filtered if m["title"] == selected),
            "title": selected,
            "genres": next(m["genres"] for m in filtered if m["title"] == selected),
            "rating": rate
        }])
        if os.path.exists(pr_path):
            existing = pd.read_csv(pr_path)
            existing = pd.concat([existing, df], ignore_index=True)
            existing.to_csv(pr_path, index=False)
        else:
            df.to_csv(pr_path, index=False)
        st.success("Оценка сохранена.")
        if len(pd.read_csv(pr_path)) >= 3:
            retrain_model()
        st.rerun()

# ---------------------- ОСНОВНОЙ ИНТЕРФЕЙС ----------------------
def main():
    apply_modern_theme()
    st.title("Что сегодня смотрим?")
    if not check_api_connection():
        st.error("API недоступен.")
        return

    st.markdown("### Меню навигации")
    nav_cols = st.columns(5)
    pages = ["Топ популярных", "По жанру", "По контенту", "Коллаборативные", "Персональные"]
    keys = ["top", "genre", "content", "collab", "personal"]

    for i, (name, key) in enumerate(zip(pages, keys)):
        if nav_cols[i].button(name, use_container_width=True):
            st.session_state.page = key

    page = st.session_state.get("page", "top")

    st.divider()

    if page == "top":
        n = st.slider("Количество фильмов", 1, 20, 10)
        if st.button("Показать топ"):
            data = get_top_movies(n)
            display_movies(data)

    elif page == "genre":
        genre = st.selectbox("Жанр", ["Action", "Comedy", "Drama", "Sci-Fi", "Horror"])
        n = st.slider("Количество фильмов", 1, 20, 10)
        if st.button("Показать по жанру"):
            display_movies(get_recommendations_by_genre(genre, n))

    elif page == "content":
        title = st.text_input("Введите название фильма")
        n = st.slider("Количество похожих", 1, 20, 10)
        if st.button("Найти похожие") and title:
            display_movies(get_content_recommendations(title, n))

    elif page == "collab":
        user = st.number_input("ID пользователя", 1, 610, 1)
        n = st.slider("Количество фильмов", 1, 20, 10)
        if st.button("Рекомендовать"):
            display_movies(get_collaborative_recommendations(user, n), True)
        if st.button("Показать оценки"):
            data = get_user_ratings(user)
            if data and "ratings" in data:
                display_ratings_table(pd.DataFrame(data["ratings"]))
            else:
                st.info("Нет данных об оценках.")

    elif page == "personal":
        if not os.path.exists(os.path.join(BASE_DIR, "personal_ratings_detailed.csv")):
            st.info("Нет персональных оценок — создайте первые:")
            create_initial_ratings()
        else:
            tabs = st.tabs(["Мои оценки", "Добавить", "Рекомендации"])
            with tabs[0]:
                df = pd.read_csv(os.path.join(BASE_DIR, "personal_ratings_detailed.csv"))
                display_ratings_table(df)
            with tabs[1]:
                add_single_rating()
            with tabs[2]:
                n = st.slider("Количество рекомендаций", 1, 20, 10)
                if st.button("Получить рекомендации"):
                    display_movies(get_personal_recommendations(n), True)

# ---------------------- ЗАПУСК ----------------------
if __name__ == "__main__":
    main()
