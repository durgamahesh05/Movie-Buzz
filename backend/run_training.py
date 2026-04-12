"""
run_training.py  -  MongoDB-first recommender bootstrap/training entrypoint
"""

from __future__ import annotations

import argparse

import recommender
import train_benchmark_models


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load MovieBuzz data into MongoDB and build recommender artifacts"
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Do not import MovieLens CSV data into MongoDB",
    )
    parser.add_argument(
        "--skip-engine",
        action="store_true",
        help="Do not warm the recommender engine after data load",
    )
    parser.add_argument(
        "--max-movies",
        type=int,
        default=recommender.IMPORT_MAX_MOVIES,
        help="Maximum number of popular movies to import into MongoDB before training (0 imports all selected movies)",
    )
    parser.add_argument(
        "--max-ratings",
        type=int,
        default=recommender.IMPORT_MAX_RATINGS,
        help="Maximum number of ratings linked to the selected movies to import into MongoDB before training (0 imports all matched ratings)",
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
        help="Fetch OMDb metadata in batches after loading data",
    )
    parser.add_argument(
        "--poster-limit",
        type=int,
        default=None,
        help="Optional maximum number of OMDb cache entries to prefetch",
    )
    parser.add_argument(
        "--poster-batch-size",
        type=int,
        default=recommender.OMDB_PREFETCH_BATCH_SIZE,
        help="Number of movies to process per OMDb batch",
    )
    parser.add_argument(
        "--train-benchmarks",
        action="store_true",
        help="Train LightGBM, CatBoost, Logistic Regression, and Random Forest benchmarks",
    )
    parser.add_argument(
        "--benchmark-train-rows",
        type=int,
        default=train_benchmark_models.DEFAULT_TRAIN_ROWS,
        help="Sampled training rows for the benchmark suite",
    )
    parser.add_argument(
        "--benchmark-val-rows",
        type=int,
        default=train_benchmark_models.DEFAULT_VAL_ROWS,
        help="Sampled validation rows for the benchmark suite",
    )
    parser.add_argument(
        "--benchmark-test-rows",
        type=int,
        default=train_benchmark_models.DEFAULT_TEST_ROWS,
        help="Sampled test rows for the benchmark suite",
    )
    parser.add_argument(
        "--benchmark-top-genres",
        type=int,
        default=train_benchmark_models.DEFAULT_TOP_GENRES,
        help="Number of top genres to encode as benchmark features",
    )
    args = parser.parse_args()

    recommender.log.info("Initialising MongoDB collections …")
    recommender.init_db()

    if not args.skip_load:
        recommender.log.info("Loading MovieLens data into MongoDB if needed …")
        recommender.load_ml25m_to_db(
            max_movies=args.max_movies,
            max_ratings=args.max_ratings,
            include_genome=args.include_genome,
            persist_tags=args.persist_tags,
            persist_rating_timestamps=args.persist_rating_timestamps,
        )

    if not args.skip_engine:
        recommender.log.info("Building recommender engine …")
        recommender.RecommenderEngine.reset()
        recommender.RecommenderEngine.get()

    if args.prefetch_posters:
        summary = recommender.prefetch_omdb_cache(
            limit=args.poster_limit,
            batch_size=args.poster_batch_size,
        )
        recommender.log.info("OMDb prefetch summary: %s", summary)

    if args.train_benchmarks:
        recommender.log.info("Training benchmark model suite …")
        report = train_benchmark_models.train_benchmark_suite(
            train_rows=args.benchmark_train_rows,
            val_rows=args.benchmark_val_rows,
            test_rows=args.benchmark_test_rows,
            seed=42,
            chunksize=train_benchmark_models.DEFAULT_CHUNK_SIZE,
            label_threshold=train_benchmark_models._env_float(
                "MOVIEBUZZ_POSITIVE_RATING_THRESHOLD",
                4.0,
            ),
            top_genres=args.benchmark_top_genres,
        )
        recommender.log.info(
            "Benchmark training completed. Models: %s",
            ", ".join(sorted((report.get("metrics") or {}).keys())),
        )


if __name__ == "__main__":
    main()
