"""
bootstrap_backend.py - one-shot backend bootstrap helper

Loads MovieLens CSV data into MongoDB, trains cached models from streamed samples,
and optionally prefetches OMDb poster/plot metadata in chunks.
"""

from __future__ import annotations

import argparse

import recommender


def main():
    parser = argparse.ArgumentParser(description="Bootstrap MovieBuzz backend data and model caches")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Only initialise/load data; do not build or train the recommender engine",
    )
    parser.add_argument(
        "--max-movies",
        type=int,
        default=recommender.IMPORT_MAX_MOVIES,
        help="Maximum number of popular movies to import into MongoDB (0 imports all selected movies)",
    )
    parser.add_argument(
        "--max-ratings",
        type=int,
        default=recommender.IMPORT_MAX_RATINGS,
        help="Maximum number of ratings linked to the selected movies to import into MongoDB (0 imports all matched ratings)",
    )
    parser.add_argument(
        "--include-genome",
        action="store_true",
        help="Also import genome_scores for the selected movie subset",
    )
    parser.add_argument(
        "--persist-tags",
        action="store_true",
        help="Also persist the filtered tags collection to MongoDB",
    )
    parser.add_argument(
        "--persist-rating-timestamps",
        action="store_true",
        help="Also persist the rating_timestamps collection to MongoDB",
    )
    parser.add_argument(
        "--prefetch-posters",
        action="store_true",
        help="Fetch OMDb poster/plot metadata in batches after data is loaded",
    )
    parser.add_argument(
        "--poster-limit",
        type=int,
        default=None,
        help="Optional maximum number of movies to prefetch from OMDb",
    )
    parser.add_argument(
        "--poster-batch-size",
        type=int,
        default=recommender.OMDB_PREFETCH_BATCH_SIZE,
        help="Number of movies to process per OMDb batch",
    )
    args = parser.parse_args()

    recommender.log.info("Initialising DB …")
    recommender.init_db()
    recommender.log.info("Loading dataset into MongoDB if needed …")
    recommender.load_ml25m_to_db(
        max_movies=args.max_movies,
        max_ratings=args.max_ratings,
        include_genome=args.include_genome,
        persist_tags=args.persist_tags,
        persist_rating_timestamps=args.persist_rating_timestamps,
    )

    if not args.skip_train:
        recommender.log.info("Building recommender engine (loads or trains cached models) …")
        recommender.RecommenderEngine.reset()
        recommender.RecommenderEngine.get()

    if args.prefetch_posters:
        summary = recommender.prefetch_omdb_cache(
            limit=args.poster_limit,
            batch_size=args.poster_batch_size,
        )
        recommender.log.info("OMDb prefetch summary: %s", summary)


if __name__ == "__main__":
    main()
