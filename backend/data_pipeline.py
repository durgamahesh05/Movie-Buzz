"""
data_pipeline.py  -  MongoDB data bootstrap helper

This imports MovieLens data into MongoDB and can optionally warm the
recommender engine so cached artifacts are ready.
"""

from __future__ import annotations

import argparse

import recommender


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load MovieBuzz catalog data into MongoDB and optionally warm the engine"
    )
    parser.add_argument(
        "--skip-engine",
        action="store_true",
        help="Only load MongoDB data; do not build the recommender engine",
    )
    args = parser.parse_args()

    recommender.log.info("Initialising MongoDB collections …")
    recommender.init_db()
    recommender.log.info("Loading MovieLens data into MongoDB if needed …")
    recommender.load_ml25m_to_db()

    if not args.skip_engine:
        recommender.log.info("Building recommender engine …")
        recommender.RecommenderEngine.reset()
        recommender.RecommenderEngine.get()


if __name__ == "__main__":
    main()
