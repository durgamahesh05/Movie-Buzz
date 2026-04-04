import os
from fastapi.testclient import TestClient
from unittest.mock import patch

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
@patch("recommender.get_db")
def test_search_movies_uses_lightweight_fallback_when_db_results_are_empty(
    mock_get_db,
    mock_search_movies_from_lightweight_catalog,
):
    mock_conn = mock_get_db.return_value.__enter__.return_value
    mock_conn.execute.return_value.fetchall.return_value = []
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
    mock_search_movies_from_lightweight_catalog.assert_called_once_with(
        "harry potter",
        limit=5,
        exclude_keys=set(),
    )


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
    movie_payload = [{"title": "New Movie", "genres": "Action", "rating": 5.0, "year": "2023", "poster": ""}]
    response = client.post("/admin/movies/manual", json=movie_payload)
    
    assert response.status_code == 200
    assert response.json() == {"inserted": 1, "status": "ok"}
    mock_add_movies.assert_called_once()


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


@patch("recommender._curated_movies_for_genre", return_value=[])
@patch("recommender._hydrate_visible_movie_posters", side_effect=lambda movies, max_items: movies)
@patch("recommender._row_to_movie_payload")
@patch("recommender.get_db")
def test_get_home_movies_prioritises_admin_movies(
    mock_get_db,
    mock_row_to_movie_payload,
    mock_hydrate_visible_movie_posters,
    mock_curated_movies_for_genre,
):
    recommender._HOME_MOVIES_CACHE.clear()

    mock_conn = mock_get_db.return_value.__enter__.return_value
    mock_conn.execute.return_value.fetchall.return_value = [
        {
            "movieId": 11,
            "title": "Catalog Movie (2024)",
            "genres": "Action",
            "avg_rating": 9.1,
            "num_ratings": 100,
            "trending_score": 25.0,
            "poster": "",
            "source": "catalog",
        },
        {
            "movieId": -22,
            "title": "Admin Movie (2025)",
            "genres": "Drama",
            "avg_rating": 6.2,
            "num_ratings": 1,
            "trending_score": 0.0,
            "poster": "",
            "source": "admin",
        },
    ]
    mock_row_to_movie_payload.side_effect = lambda row: {
        "movie_key": str(row["movieId"]),
        "title": row["title"],
    }

    recommender.get_home_movies(limit=2)
    executed_query = mock_conn.execute.call_args[0][0]

    assert "CASE WHEN source = 'admin' THEN 0 ELSE 1 END" in executed_query


@patch("recommender._load_user_preference_context", return_value={"preferred_genres": ["action"]})
@patch("recommender._curated_movies_for_genre", return_value=[])
@patch("recommender._hydrate_visible_movie_posters", side_effect=lambda movies, max_items: movies)
@patch("recommender.get_db")
def test_get_home_movies_expands_candidate_pool_for_preferences(
    mock_get_db,
    mock_hydrate_visible_movie_posters,
    mock_curated_movies_for_genre,
    mock_load_user_preference_context,
):
    recommender._HOME_MOVIES_CACHE.clear()

    mock_conn = mock_get_db.return_value.__enter__.return_value
    mock_conn.execute.return_value.fetchall.return_value = []

    recommender.get_home_movies(limit=10, user_email="test@example.com")
    executed_params = mock_conn.execute.call_args[0][1]

    assert executed_params[-1] == 160
