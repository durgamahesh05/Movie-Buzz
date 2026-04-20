import os
from fastapi.testclient import TestClient
from unittest.mock import patch
from unittest.mock import MagicMock

os.environ["MOVIEBUZZ_SKIP_STARTUP"] = "1"

from app import app
from auth_routes import PASSWORD_POLICY_MESSAGE
import recommender

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "MovieBuzz backend v3 running" in data["status"]
    assert "models" in data
    assert "signals" in data

@patch("app.search_movies")
def test_search_success(mock_search_movies):
    mock_search_movies.return_value = [{"title": "Inception", "year": "2010"}]
    response = client.get("/search?q=Inception")
    
    assert response.status_code == 200
    assert response.json() == [{"title": "Inception", "year": "2010"}]
    mock_search_movies.assert_called_once_with("Inception", limit=50)

def test_search_missing_query():
    response = client.get("/search")
    assert response.status_code == 422


@patch("recommender._candidate_to_movie_payload", side_effect=lambda candidate: {
    "movie_key": candidate["movie_key"],
    "movie_id": candidate["movie_id"],
    "title": candidate["title"],
    "source": candidate.get("source"),
})
@patch("recommender._lightweight_catalog_movies")
def test_lightweight_catalog_search_matches_partial_titles(
    mock_lightweight_catalog_movies,
    mock_candidate_to_movie_payload,
):
    mock_lightweight_catalog_movies.return_value = [
        recommender._movie_search_candidate(
            title="Harry Potter and the Chamber of Secrets (2002)",
            genres="Adventure Fantasy",
            movie_id=5816,
            source="ml25m_csv",
        ),
        recommender._movie_search_candidate(
            title="The Matrix (1999)",
            genres="Action Sci-Fi",
            movie_id=2571,
            source="ml25m_csv",
        ),
    ]

    results = recommender._search_movies_from_lightweight_catalog(
        "harry potter",
        limit=5,
    )

    assert len(results) == 1
    assert results[0]["title"] == "Harry Potter and the Chamber of Secrets (2002)"
    assert results[0]["source"] == "ml25m_csv"


@patch("recommender._search_movies_from_lightweight_catalog")
def test_search_movies_uses_lightweight_fallback_when_db_results_are_empty(
    mock_search_movies_from_lightweight_catalog,
):
    mock_search_movies_from_lightweight_catalog.return_value = [
        {
            "movie_key": f"harry-potter-{index}",
            "title": f"Harry Potter {index}",
        }
        for index in range(5)
    ]

    results = recommender.search_movies("harry potter", limit=5)

    assert len(results) == 5
    assert all("Harry Potter" in movie["title"] for movie in results)
    mock_search_movies_from_lightweight_catalog.assert_called_once_with("harry potter", limit=5)


@patch("recommender._lightweight_catalog_movies")
@patch(
    "recommender._candidate_to_movie_payload",
    side_effect=lambda candidate: {
        "movie_key": candidate["movie_key"],
        "movie_id": candidate["movie_id"],
        "title": candidate["title"],
    },
)
@patch("recommender.get_collection")
def test_list_admin_movies_queries_storage_without_full_catalog_scan(
    mock_get_collection,
    mock_candidate_to_movie_payload,
    mock_lightweight_catalog_movies,
):
    recommender._ADMIN_CATALOG_GENRES_CACHE.clear()
    recommender._LIGHTWEIGHT_TITLE_CATALOG_CACHE.clear()

    movies_collection = MagicMock()
    movies_collection.count_documents.return_value = 2
    movies_collection.distinct.return_value = ["Action Comedy", "Drama"]
    movies_collection.aggregate.return_value = [
        {
            "movieId": -3,
            "title": "Admin Pick (2026)",
            "genres": "Action Comedy",
            "avg_rating": 7.4,
            "num_ratings": 1,
            "trending_score": 0.0,
            "poster": "",
            "source": "admin",
        }
    ]
    mock_get_collection.return_value = movies_collection

    try:
        payload = recommender.list_admin_movies(limit=1, offset=0)
    finally:
        recommender._ADMIN_CATALOG_GENRES_CACHE.clear()
        recommender._LIGHTWEIGHT_TITLE_CATALOG_CACHE.clear()

    assert payload["total"] == 2
    assert payload["limit"] == 1
    assert payload["offset"] == 0
    assert payload["has_more"] is True
    assert payload["genres"] == ["Action", "Comedy", "Drama"]
    assert payload["items"][0]["movie_key"] == "admin-pick-2026"
    assert payload["items"][0]["movie_id"] == -3
    assert payload["items"][0]["title"] == "Admin Pick (2026)"
    assert payload["items"][0]["source"] == "admin"
    assert payload["items"][0]["source_label"] == "Admin"
    assert payload["items"][0]["can_delete"] is True
    assert payload["items"][0]["imdb_rating"] == "7.4"
    assert payload["items"][0]["rating"] == 7.4
    movies_collection.count_documents.assert_called_once_with({})
    movies_collection.aggregate.assert_called_once()
    movies_collection.distinct.assert_called_once_with("genres")
    mock_lightweight_catalog_movies.assert_not_called()


@patch("recommender._apply_preference_ranking", side_effect=lambda movies, user_email, limit: movies[:limit])
@patch("recommender._user_feedback_movie_ids", return_value=({22}, set()))
@patch("recommender._lightweight_catalog_movies")
@patch("recommender._search_movies_from_lightweight_catalog")
def test_recommend_from_database_uses_saved_feedback_for_lightweight_results(
    mock_search_movies_from_lightweight_catalog,
    mock_lightweight_catalog_movies,
    mock_user_feedback_movie_ids,
    mock_apply_preference_ranking,
):
    anchor = recommender._movie_search_candidate(
        title="Quiet Drama (2024)",
        genres="Drama",
        movie_id=11,
        source="admin",
    )
    liked_candidate = recommender._movie_search_candidate(
        title="Solar Storm (2024)",
        genres="Sci-Fi",
        avg_rating=5.0,
        movie_id=22,
        source="admin",
    )
    same_genre_candidate = recommender._movie_search_candidate(
        title="Family Tale (2020)",
        genres="Drama",
        avg_rating=0.0,
        movie_id=33,
        source="catalog",
    )

    mock_search_movies_from_lightweight_catalog.return_value = [anchor]
    mock_lightweight_catalog_movies.return_value = [
        anchor,
        liked_candidate,
        same_genre_candidate,
    ]

    results = recommender._recommend_from_database(
        "Quiet Drama",
        top_n=2,
        user_email="tester@example.com",
    )

    assert results["resolved_title"] == "Quiet Drama"
    assert [movie["title"] for movie in results["results"]] == [
        "Solar Storm (2024)",
        "Family Tale (2020)",
    ]
    mock_user_feedback_movie_ids.assert_called_once_with("tester@example.com")
    mock_apply_preference_ranking.assert_called_once()


@patch("app.recommend_movies")
def test_recommend_success(mock_recommend_movies):
    mock_recommend_movies.return_value = {"resolved_title": "The Matrix", "results": []}
    response = client.get("/recommend?title=Matrix&user_id=42&mood=action")
    
    assert response.status_code == 200
    assert response.json()["resolved_title"] == "The Matrix"
    mock_recommend_movies.assert_called_once_with(
        "Matrix",
        42,
        mood="action",
        top_n=50,
        user_email=None,
    )


@patch("trailer_router._cache_is_fresh", return_value=True)
@patch("trailer_router._trailer_cache")
def test_trailer_route_returns_cached_negative_result(
    mock_trailer_cache,
    mock_cache_is_fresh,
):
    cache_collection = MagicMock()
    cache_collection.find_one.return_value = {
        "title": "No Trailer Movie",
        "year": "2024",
        "video_id": None,
        "fetched_at": "2026-04-14T00:00:00",
    }
    mock_trailer_cache.return_value = cache_collection

    response = client.get("/api/trailer/321")

    assert response.status_code == 200
    assert response.json() == {
        "movie_id": 321,
        "title": "No Trailer Movie",
        "year": "2024",
        "video_id": None,
        "embed_url": None,
        "found": False,
    }
    mock_cache_is_fresh.assert_called_once()

def test_list_moods():
    response = client.get("/moods")
    assert response.status_code == 200
    data = response.json()
    assert "moods" in data
    assert "happy" in data["moods"]
    assert "scared" in data["moods"]

@patch("app.browse_mood")
def test_mood_browse_valid(mock_browse_mood):
    mock_browse_mood.return_value = [{"title": "Happy Gilmore"}]
    response = client.get("/mood/happy")
    
    assert response.status_code == 200
    assert response.json() == [{"title": "Happy Gilmore"}]
    mock_browse_mood.assert_called_once_with("happy")

def test_mood_browse_invalid():
    response = client.get("/mood/angry")
    assert response.status_code == 400
    assert "Unknown mood" in response.json()["detail"]

@patch("app.record_feedback")
def test_feedback_valid(mock_record_feedback):
    mock_record_feedback.return_value = True
    response = client.post("/feedback", json={"user_id": "user123", "movie_id": 101, "feedback": "like"})
    
    assert response.status_code == 200
    assert response.json()["status"] == "recorded"
    mock_record_feedback.assert_called_once_with("user123", 101, "like")

@patch("app.record_feedback")
def test_feedback_invalid(mock_record_feedback):
    mock_record_feedback.return_value = False
    response = client.post("/feedback", json={"user_id": "user123", "movie_id": 101, "feedback": "unknown"})
    
    assert response.status_code == 400
    assert "feedback must be" in response.json()["detail"]

@patch("app.add_movies_to_db")
def test_admin_add_manual(mock_add_movies):
    mock_add_movies.return_value = 1
    movie_payload = [{
        "title": "New Movie",
        "genres": "Action",
        "rating": 5.0,
        "year": "2023",
        "poster": "",
        "youtube_link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    }]
    response = client.post("/admin/movies/manual", json=movie_payload)
    
    assert response.status_code == 200
    assert response.json() == {"inserted": 1, "status": "ok"}
    mock_add_movies.assert_called_once_with(movie_payload)


@patch.object(recommender.RecommenderEngine, "_fetch_omdb", return_value={})
@patch.object(recommender.RecommenderEngine, "reset")
@patch("recommender._next_admin_movie_id", return_value=-1)
@patch("recommender.get_collection")
def test_add_movies_to_db_preserves_supplied_youtube_url(
    mock_get_collection,
    _mock_next_admin_movie_id,
    _mock_reset,
    _mock_fetch_omdb,
):
    movies_collection = MagicMock()
    mock_get_collection.return_value = movies_collection
    movie_payload = [{
        "title": "New Movie",
        "genres": "Action",
        "rating": 5.0,
        "year": "2023",
        "poster": "",
        "youtube_link": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    }]

    inserted = recommender.add_movies_to_db(movie_payload)

    assert inserted == 1
    mock_get_collection.assert_called_once_with("movies")
    inserted_rows = movies_collection.insert_many.call_args.args[0]
    assert inserted_rows[0]["youtube_link"] == movie_payload[0]["youtube_link"]


@patch("auth_routes.update_preferences")
@patch("auth_routes.get_preferences")
@patch("auth_routes.find_one")
def test_save_preferences_clears_values_when_fields_explicitly_blank(
    mock_find_one,
    mock_get_preferences,
    mock_update_preferences,
):
    mock_find_one.return_value = {
        "email": "test@example.com",
        "name": "Test User",
    }
    mock_get_preferences.return_value = {
        "age": 28,
        "preferred_genres": ["Action", "Drama"],
        "preferred_moods": ["happy"],
    }

    response = client.post(
        "/auth/preferences",
        json={
            "email": "test@example.com",
            "age": "",
            "preferred_genres": [],
            "preferred_moods": [],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "msg": "Preferences saved",
        "name": "Test User",
        "email": "test@example.com",
        "age": None,
        "preferred_genres": [],
        "preferred_moods": [],
    }
    mock_update_preferences.assert_called_once_with(
        "test@example.com",
        age=None,
        preferred_genres=[],
        preferred_moods=[],
    )


def test_signup_rejects_weak_password():
    response = client.post(
        "/auth/signup",
        json={
            "name": "Test User",
            "email": "test@example.com",
            "password": "weak1!",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"success": False, "msg": PASSWORD_POLICY_MESSAGE}


@patch("auth_routes.find_one")
def test_forgot_password_reset_rejects_weak_password(mock_find_one):
    mock_find_one.return_value = {"email": "test@example.com"}

    response = client.post(
        "/auth/forgot-password/reset",
        json={
            "email": "test@example.com",
            "otp": "123456",
            "new_password": "weak1!",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"success": False, "msg": PASSWORD_POLICY_MESSAGE}


@patch("auth_routes.bcrypt.checkpw", return_value=True)
@patch(
    "auth_routes.get_preferences",
    return_value={"age": None, "preferred_genres": [], "preferred_moods": []},
)
@patch("auth_routes.find_one_by_login_identifier")
def test_login_allows_admin_username_identifier(
    mock_find_one_by_login_identifier,
    mock_get_preferences,
    mock_checkpw,
):
    mock_find_one_by_login_identifier.return_value = {
        "name": "Admin 61",
        "email": "admin61@moviebuzz.in",
        "password": "hashed-password",
        "verified": True,
        "role": "admin",
    }

    response = client.post(
        "/auth/login",
        json={
            "email": "admin_61",
            "password": "Moviebuzz@61",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "msg": "Login successful",
        "name": "Admin 61",
        "email": "admin61@moviebuzz.in",
        "role": "admin",
        "age": None,
        "preferred_genres": [],
        "preferred_moods": [],
    }
    mock_find_one_by_login_identifier.assert_called_once_with("admin_61")
    mock_get_preferences.assert_called_once_with("admin61@moviebuzz.in")
    mock_checkpw.assert_called_once()


@patch("recommender._curated_movies_for_genre", return_value=[])
@patch("recommender._hydrate_visible_movie_posters", side_effect=lambda movies, max_items: movies)
@patch("recommender._candidate_to_movie_payload")
@patch("recommender._lightweight_catalog_movies")
def test_get_home_movies_prioritises_admin_movies(
    mock_lightweight_catalog_movies,
    mock_candidate_to_movie_payload,
    mock_hydrate_visible_movie_posters,
    mock_curated_movies_for_genre,
):
    recommender._HOME_MOVIES_CACHE.clear()

    mock_lightweight_catalog_movies.return_value = [
        {
            "movie_id": 11,
            "title": "Catalog Movie (2024)",
            "genres": "Action",
            "rating": 9.1,
            "num_ratings": 100,
            "trending_score": 25.0,
            "poster": "",
            "source": "catalog",
        },
        {
            "movie_id": -22,
            "title": "Admin Movie (2025)",
            "genres": "Drama",
            "rating": 6.2,
            "num_ratings": 1,
            "trending_score": 0.0,
            "poster": "",
            "source": "admin",
        },
    ]
    mock_candidate_to_movie_payload.side_effect = lambda row: {
        "movie_key": str(row["movie_id"]),
        "title": row["title"],
    }

    results = recommender.get_home_movies(limit=2)

    assert results[0]["title"] == "Admin Movie (2025)"
    assert results[1]["title"] == "Catalog Movie (2024)"


@patch("recommender._load_user_preference_context", return_value={"preferred_genres": ["action"]})
@patch("recommender._curated_movies_for_genre", return_value=[])
@patch("recommender._hydrate_visible_movie_posters", side_effect=lambda movies, max_items: movies)
@patch("recommender._candidate_to_movie_payload")
@patch("recommender._lightweight_catalog_movies")
def test_get_home_movies_expands_candidate_pool_for_preferences(
    mock_lightweight_catalog_movies,
    mock_candidate_to_movie_payload,
    mock_hydrate_visible_movie_posters,
    mock_curated_movies_for_genre,
    mock_load_user_preference_context,
):
    recommender._HOME_MOVIES_CACHE.clear()

    mock_lightweight_catalog_movies.return_value = [
        {
            "movie_key": f"movie-{index}",
            "movie_id": index,
            "title": f"Movie {index} (2024)",
            "clean_title": f"Movie {index}",
            "year": "2024",
            "genres": "Action",
            "rating": 5.0,
            "num_ratings": 10,
            "trending_score": 1.0,
            "poster": "",
            "source": "catalog",
        }
        for index in range(1, 201)
    ]
    mock_candidate_to_movie_payload.side_effect = lambda row: {
        "movie_key": row["movie_key"],
        "movie_id": row["movie_id"],
        "title": row["title"],
        "genres": row["genres"],
        "rating": row["rating"],
        "trending_score": row["trending_score"],
        "poster": row["poster"],
    }

    recommender.get_home_movies(limit=10, user_email="test@example.com")

    assert mock_candidate_to_movie_payload.call_count == 160
