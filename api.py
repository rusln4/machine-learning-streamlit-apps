from fastapi import FastAPI, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = "models"                  
                                 
STATE = {
    "movies": None,
    "ratings": None,
    "movie_stats": None,
                      
    "tfidf_vectorizer": None,
    "content_similarity": None,                                
    "movie_to_idx": None,
    "idx_to_movie": None,
                                 
    "nmf_model": None,                                      
    "W": None,                                         
    "H": None,                                          
    "user_movie_matrix": None,                                  
    "user_to_idx": None,
    "idx_to_user": None,
                                                      
    "personal_profile": None,
}

# Загрузка моделей и данных
def load_artifacts():
                    
    movies_p = os.path.join(MODELS_DIR, "movies.pkl")
    ratings_p = os.path.join(MODELS_DIR, "ratings.pkl")
    movie_stats_p = os.path.join(MODELS_DIR, "movie_stats.pkl")

    # Фильмы
    if os.path.exists(movies_p):
        STATE["movies"] = pd.read_pickle(movies_p)
    else:
        STATE["movies"] = pd.DataFrame(columns=["movieId", "title", "genres"])

    # Рейтинги
    if os.path.exists(ratings_p):
        STATE["ratings"] = pd.read_pickle(ratings_p)
    else:
        STATE["ratings"] = pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])

    # Статистика фильмов
    if os.path.exists(movie_stats_p):
        STATE["movie_stats"] = pd.read_pickle(movie_stats_p)
    else:
        # Если статистики нет — создаём вручную                                                           
        ratings = STATE["ratings"]
        movies = STATE["movies"]
        if not ratings.empty and not movies.empty:
            stats = (
                ratings.groupby("movieId")
                .agg(rating_count=("rating", "count"), rating_mean=("rating", "mean"), rating_std=("rating", "std"))
                .reset_index()
            ).merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")
                                                         
            # Вычисляем взвешенный рейтинг IMDb
            m = stats["rating_count"].quantile(0.9) if not stats.empty else 0
            C = stats["rating_mean"].mean() if not stats.empty else 0
            stats["weighted_rating"] = (stats["rating_count"] / (stats["rating_count"] + m)) * stats["rating_mean"] + (
                m / (stats["rating_count"] + m)
            ) * C
            STATE["movie_stats"] = stats.sort_values("weighted_rating", ascending=False)
        else:
            STATE["movie_stats"] = pd.DataFrame(columns=["movieId", "title", "genres", "weighted_rating", "rating_count", "rating_mean"])

                      
    tfidf_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    cs_npz_path = os.path.join(MODELS_DIR, "content_similarity.npz")
    cs_pkl_path = os.path.join(MODELS_DIR, "content_similarity.pkl")
    movie_mappings_p = os.path.join(MODELS_DIR, "movie_mappings.pkl")

    if os.path.exists(tfidf_path):
        with open(tfidf_path, "rb") as f:
            STATE["tfidf_vectorizer"] = pickle.load(f)

    if os.path.exists(cs_npz_path):
        cs = np.load(cs_npz_path)
        STATE["content_similarity"] = cs["content_similarity"]
    elif os.path.exists(cs_pkl_path):
        with open(cs_pkl_path, "rb") as f:
            STATE["content_similarity"] = pickle.load(f)

    if os.path.exists(movie_mappings_p):
        with open(movie_mappings_p, "rb") as f:
            mm = pickle.load(f)
        STATE["movie_to_idx"] = mm.get("movie_to_idx")
        STATE["idx_to_movie"] = mm.get("idx_to_movie")

    # === Collaborative модели (NMF) ===
    collab_p = os.path.join(MODELS_DIR, "collaborative_model.pkl")
    umm_pkl = os.path.join(MODELS_DIR, "user_movie_matrix.pkl")
    user_mappings_p = os.path.join(MODELS_DIR, "user_mappings.pkl")

    if os.path.exists(collab_p):
        with open(collab_p, "rb") as f:
            obj = pickle.load(f)
                                                                 
        if isinstance(obj, dict):
            STATE["nmf_model"] = obj.get("model") or obj.get("nmf_model")
            STATE["W"] = obj.get("W")
            STATE["H"] = obj.get("H")
        else:
            STATE["nmf_model"] = obj

    if os.path.exists(umm_pkl):
        with open(umm_pkl, "rb") as f:
            STATE["user_movie_matrix"] = pickle.load(f)             
    else:
        STATE["user_movie_matrix"] = pd.DataFrame()

    if os.path.exists(user_mappings_p):
        with open(user_mappings_p, "rb") as f:
            um = pickle.load(f)
        STATE["user_to_idx"] = um.get("user_to_idx")
        STATE["idx_to_user"] = um.get("idx_to_user")

                                                             
    if STATE["H"] is None and STATE["nmf_model"] is not None and hasattr(STATE["nmf_model"], "components_"):
        STATE["H"] = STATE["nmf_model"].components_

                                                                                   
    if STATE["W"] is None and STATE["nmf_model"] is not None and isinstance(STATE["user_movie_matrix"], pd.DataFrame) and not STATE["user_movie_matrix"].empty:
        try:
                                        
            X = np.maximum(STATE["user_movie_matrix"].values, 0)
            STATE["W"] = STATE["nmf_model"].transform(X)
        except Exception as e:
            print(f"Не удалось вычислить W через transform: {e}")

                         
   
   
load_artifacts()


@app.get("/")
def home():
    return {
        "message": "API рекомендательной системы фильмов",
        "version": "1.0",
        "endpoints": {
            "/api/recommendations/top-movies": "Топ популярных фильмов",
            "/api/recommendations/genre/{genre}": "Рекомендации по жанру",
            "/api/recommendations/similar/{movie_title}": "Рекомендации по контенту",
            "/api/recommendations/collaborative/{user_id}": "Коллаборативные рекомендации",
            "/api/recommendations/personal": "Персональные рекомендации",
            "/api/user/{user_id}/ratings": "Рейтинги пользователя"
        }
    }

# Топ популярных фильмов
@app.get("/api/recommendations/top-movies")
@app.get("/api/top-movies")
def get_top_movies(n: int = Query(10)):
    try:
        stats = STATE["movie_stats"]
        if stats is None or stats.empty:
            return []

        top_movies = stats.head(n)[["movieId", "title", "genres", "weighted_rating", "rating_count", "rating_mean"]]
        movies_list = []
        for _, movie in top_movies.iterrows():
            movies_list.append({
                "movie_id": int(movie["movieId"]),
                "title": movie["title"],
                "genres": movie["genres"],
                "weighted_rating": float(movie["weighted_rating"]),
                "rating_count": int(movie["rating_count"]),
                "rating_mean": float(movie["rating_mean"]),
            })
        return movies_list
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Рекомендации по жанру
@app.get("/api/recommendations/genre/{genre}")
def get_recommendations_by_genre(genre: str = Path(...), n: int = Query(10)):
    try:
        stats = STATE["movie_stats"]
        if stats is None or stats.empty:
            return []

        genre_movies = stats[stats["genres"].str.contains(genre, case=False, na=False)]
        if genre_movies.empty:
            return []

        recs = genre_movies.sort_values("weighted_rating", ascending=False).head(n)
        movies_list = []
        for _, movie in recs.iterrows():
            movies_list.append({
                "movie_id": int(movie["movieId"]),
                "title": movie["title"],
                "genres": movie["genres"],
                "weighted_rating": float(movie["weighted_rating"]),
                "rating_count": int(movie["rating_count"]),
                "rating_mean": float(movie["rating_mean"]),                                       
            })
        return movies_list
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Похожие фильмы (Content-Based)
@app.get("/api/recommendations/similar/{movie_title}")
def get_content_recommendations(movie_title: str = Path(...), n: int = Query(10)):
    try:
        movies = STATE["movies"]
        S = STATE["content_similarity"]
        movie_to_idx = STATE["movie_to_idx"]
        idx_to_movie = STATE["idx_to_movie"]

        if movies is None or movies.empty or S is None or movie_to_idx is None or idx_to_movie is None:
            return []

        matches = movies[movies["title"].str.contains(movie_title, case=False, na=False)]
        if matches.empty:
            return []

        movie_id = int(matches.iloc[0]["movieId"])
        if movie_id not in movie_to_idx:
            return []

        idx = movie_to_idx[movie_id]
        sims = S[idx]

        top_idx = np.argsort(sims)[::-1]            
        movies_list = []
        for j in top_idx:
            if j == idx:
                continue
            mid = idx_to_movie[j]
            movie_row = movies[movies["movieId"] == mid]
            if movie_row.empty:
                continue
            row = movie_row.iloc[0]
            movies_list.append({
                "movie_id": int(mid),
                "title": row["title"],
                "genres": row["genres"],
                "similarity_score": float(sims[j]),
            })
            if len(movies_list) >= n:
                break

        return movies_list
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Коллаборативные рекомендации (NMF)
@app.get("/api/recommendations/collaborative/{user_id}")
def get_collaborative_recommendations(user_id: int = Path(...), n: int = Query(10)):
    try:
        nmf_model = STATE["nmf_model"]
        W = STATE["W"]
        H = STATE["H"]
        UMM = STATE["user_movie_matrix"]
        user_to_idx = STATE["user_to_idx"]
        movies = STATE["movies"]

        if H is None or UMM is None or UMM.empty or user_to_idx is None:
            return []
        if user_id not in user_to_idx:
                                                 
            stats = STATE["movie_stats"]
            if stats is None or stats.empty:
                return []
            top_movies = stats.head(n)
            return [{
                "movie_id": int(row["movieId"]),
                "title": row["title"],
                "genres": row["genres"],
                "predicted_rating": float(row["weighted_rating"]),
            } for _, row in top_movies.iterrows()]

        user_idx = user_to_idx[user_id]
                                                                                  
        if W is None:
            x = np.maximum(UMM.values, 0)
            if nmf_model is None:
                                                       
                stats = STATE["movie_stats"]
                if stats is None or stats.empty:
                    return []
                top_movies = stats.head(n)
                return [{
                    "movie_id": int(row["movieId"]),
                    "title": row["title"],
                    "genres": row["genres"],
                    "predicted_rating": float(row["weighted_rating"]),
                } for _, row in top_movies.iterrows()]
            W_all = nmf_model.transform(x)
            W_u = W_all[user_idx]
        else:
            W_u = W[user_idx]
        preds = np.dot(W_u, H)
        rated_mask = UMM.iloc[user_idx].values > 0

                                 
        preds_filtered = preds.copy()
        preds_filtered[rated_mask] = -np.inf

        top_j = np.argsort(preds_filtered)[-n:][::-1]
        movie_ids = list(UMM.columns)

        movies_list = []
        for j in top_j:
            mid = int(movie_ids[j])
            movie_row = movies[movies["movieId"] == mid]
            if movie_row.empty:
                continue
            row = movie_row.iloc[0]
            movies_list.append({
                "movie_id": mid,
                "title": row["title"],
                "genres": row["genres"],
                "predicted_rating": float(preds[j]),
            })

        return movies_list
    except Exception as e:
        return {"status": "error", "message": str(e)}


# Персональные рекомендации                                          
@app.get("/api/recommendations/personal")
def get_personal_recommendations(n: int = Query(10)):
    try:
                                       
        if not os.path.exists("personal_ratings_detailed.csv"):
            return []

        personal_ratings = pd.read_csv("personal_ratings_detailed.csv")
                                                  
        rated_ids = set()
        for _, r in personal_ratings.iterrows():
            try:
                rated_ids.add(int(r["movieId"]))
            except Exception:
                continue
        nmf_model = STATE["nmf_model"]
        H = STATE["H"]
        UMM = STATE["user_movie_matrix"]
        movies = STATE["movies"]

        if nmf_model is None or H is None or UMM is None or UMM.empty:
            return []

        movie_id_to_col = {int(mid): i for i, mid in enumerate(UMM.columns)}

                                                                     
        W_user = STATE.get("personal_profile")
        rated_cols = [movie_id_to_col[int(r["movieId"])]
                      for _, r in personal_ratings.iterrows()
                      if int(r.get("movieId", -1)) in movie_id_to_col and float(r.get("rating", 0)) > 0]

        if W_user is None:
            user_vec = np.zeros(len(UMM.columns), dtype=np.float64)
            for _, r in personal_ratings.iterrows():
                try:
                    mid = int(r["movieId"])
                    rating = float(r["rating"])
                except Exception:
                    continue
                if mid in movie_id_to_col and rating > 0:
                    user_vec[movie_id_to_col[mid]] = rating

            if np.count_nonzero(user_vec) == 0:
                cb = content_based_personal_fallback(personal_ratings, n)
                if cb and len(cb) > 0:
                                                                       
                    cb = [m for m in cb if m.get("movie_id") not in rated_ids]
                    return cb[:n]
                stats = STATE["movie_stats"]
                if stats is None or stats.empty:
                    return []
                                                    
                filtered_stats = stats[~stats["movieId"].isin(rated_ids)]
                top_movies = filtered_stats.head(n)
                return [{
                    "movie_id": int(row["movieId"]),
                    "title": row["title"],
                    "genres": row["genres"],
                    "predicted_rating": float(row["weighted_rating"]) * 2,
                } for _, row in top_movies.iterrows()]

            W_user = nmf_model.transform(user_vec.reshape(1, -1).astype(np.float64))
                              
            STATE["personal_profile"] = W_user

        preds = np.dot(W_user, H).ravel()

                                   
        if rated_cols:
            preds[np.array(rated_cols, dtype=int)] = -np.inf

        top_j = np.argsort(preds)[-n:][::-1]
        movie_ids = list(UMM.columns)

        movies_list = []
        for j in top_j:
            mid = int(movie_ids[j])
                                                         
            if mid in rated_ids:
                continue
            movie_row = movies[movies["movieId"] == mid]
            if movie_row.empty:
                continue
            row = movie_row.iloc[0]
            movies_list.append({
                "movie_id": mid,
                "title": row["title"],
                "genres": row["genres"],
                "predicted_rating": float(preds[j]) * 2,       
                "recommendation_score": float(preds[j]),
            })

        return movies_list
    except Exception as e:
        return {"status": "error", "message": str(e)}


# Оценки пользователя            
@app.get("/api/user/{user_id}/ratings")
def get_user_ratings(user_id: int = Path(...)):
    try:
        ratings = STATE["ratings"]
        movies = STATE["movies"]
        if ratings is None or ratings.empty or movies is None or movies.empty:
            return {"status": "error", "message": "Данные не загружены"}

        user_ratings = ratings[ratings["userId"] == user_id].merge(movies, on="movieId")
        if len(user_ratings) == 0:
            return {"status": "error", "message": f"Пользователь {user_id} не найден или не имеет рейтингов"}

        ratings_list = []
        for _, rating in user_ratings.iterrows():
            ratings_list.append({
                "movieId": int(rating["movieId"]),
                "title": rating["title"],
                "genres": rating["genres"],
                "rating": float(rating["rating"]),
                "rating_10": float(rating["rating"]) * 2,
                "timestamp": int(rating["timestamp"]),
            })

        return {
            "status": "success",
            "user_id": user_id,
            "ratings": ratings_list,
            "count": len(ratings_list),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


                                                                                    
def rebuild_personal_profile():
    try:
        if not os.path.exists("personal_ratings_detailed.csv"):
            STATE["personal_profile"] = None
            return {"status": "error", "message": "Файл персональных оценок не найден", "rated_count": 0}

        personal_ratings = pd.read_csv("personal_ratings_detailed.csv")
        nmf_model = STATE["nmf_model"]
        H = STATE["H"]
        UMM = STATE["user_movie_matrix"]

        if nmf_model is None or H is None or UMM is None or UMM.empty:
            STATE["personal_profile"] = None
            return {"status": "error", "message": "Модель/артефакты не загружены", "rated_count": len(personal_ratings)}

        movie_id_to_col = {int(mid): i for i, mid in enumerate(UMM.columns)}
        user_vec = np.zeros(len(UMM.columns), dtype=np.float64)
        rated_count = 0
        for _, r in personal_ratings.iterrows():
            try:
                mid = int(r["movieId"])
                rating = float(r["rating"])
            except Exception:
                continue
            if mid in movie_id_to_col and rating > 0:
                user_vec[movie_id_to_col[mid]] = rating
                rated_count += 1

        if np.count_nonzero(user_vec) == 0:
            STATE["personal_profile"] = None
            return {"status": "error", "message": "Нет пересечения с обучающей матрицей", "rated_count": rated_count}

        W_user = nmf_model.transform(user_vec.reshape(1, -1).astype(np.float64))
        STATE["personal_profile"] = W_user
        return {"status": "success", "rated_count": rated_count}
    except Exception as e:
        STATE["personal_profile"] = None
        return {"status": "error", "message": str(e)}

def content_based_personal_fallback(personal_ratings: pd.DataFrame, n: int):
    try:
        movies = STATE["movies"]
        S = STATE["content_similarity"]
        movie_to_idx = STATE["movie_to_idx"]
        idx_to_movie = STATE["idx_to_movie"]

        if movies is None or movies.empty or S is None or movie_to_idx is None or idx_to_movie is None:
            return []

        scores = np.zeros(S.shape[0], dtype=np.float64)
        rated_ids = set()
        for _, r in personal_ratings.iterrows():
            try:
                mid = int(r["movieId"])
                rating = float(r["rating"])
            except Exception:
                continue
            if mid in movie_to_idx and rating > 0:
                rated_ids.add(mid)
                idx = movie_to_idx[mid]
                sims = S[idx]
                scores += sims * rating

        if len(rated_ids) == 0:
            return []

        for mid in rated_ids:
            idx = movie_to_idx[mid]
            scores[idx] = -np.inf

        top_idx = np.argsort(scores)[-n:][::-1]
        result = []
        for j in top_idx:
            mid = idx_to_movie[j]
            movie_row = movies[movies["movieId"] == mid]
            if movie_row.empty:
                continue
            row = movie_row.iloc[0]
            result.append({
                "movie_id": int(mid),
                "title": row["title"],
                "genres": row["genres"],
                "similarity_score": float(scores[j]),
            })
        return result
    except Exception:
        return []


# Переобучение персонального профиля                     
@app.post("/api/retrain")
def retrain_personal():
    try:
        info = rebuild_personal_profile()
        return info
    except Exception as e:
        STATE["personal_profile"] = None
        return {"status": "error", "message": str(e)}

def content_based_personal_fallback(personal_ratings: pd.DataFrame, n: int):
    try:
        movies = STATE["movies"]
        S = STATE["content_similarity"]
        movie_to_idx = STATE["movie_to_idx"]
        idx_to_movie = STATE["idx_to_movie"]

        if movies is None or movies.empty or S is None or movie_to_idx is None or idx_to_movie is None:
            return []

                                    
        scores = np.zeros(S.shape[0], dtype=np.float64)
        rated_ids = set()
        for _, r in personal_ratings.iterrows():
            try:
                mid = int(r["movieId"])
                rating = float(r["rating"])
            except Exception:
                continue
            if mid in movie_to_idx and rating > 0:
                rated_ids.add(mid)
                idx = movie_to_idx[mid]
                sims = S[idx]
                scores += sims * rating

        if len(rated_ids) == 0:
            return []

                               
        for mid in rated_ids:
            idx = movie_to_idx[mid]
            scores[idx] = -np.inf

        top_idx = np.argsort(scores)[-n:][::-1]
        result = []
        for j in top_idx:
            mid = idx_to_movie[j]
            movie_row = movies[movies["movieId"] == mid]
            if movie_row.empty:
                continue
            row = movie_row.iloc[0]
            result.append({
                "movie_id": int(mid),
                "title": row["title"],
                "genres": row["genres"],
                "similarity_score": float(scores[j]),
            })
        return result
    except Exception:
        return []


if __name__ == "__main__":
    import uvicorn
    print("Запуск API рекомендательной системы фильмов на FastAPI...")
    print("Документация доступна по адресу: http://localhost:5001/docs")
    uvicorn.run(app, host="0.0.0.0", port=5001)