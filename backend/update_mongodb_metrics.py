import os
from datetime import datetime, timezone
from pymongo import MongoClient
from dotenv import load_dotenv

# Load env file to get MONGODB_URI
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

mongo_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
kwargs = {}
if mongo_uri.startswith("mongodb+srv://"):
    kwargs["tls"] = True
    kwargs["tlsAllowInvalidCertificates"] = True

client = MongoClient(mongo_uri, **kwargs)

db = client.get_default_database("moviebuzz")
collection = db["model_metrics"]

metrics_doc = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "primary_model": "XGB",                    
    "test_ratio": 0.2,
    "models": {
        "XGB": {
            "status": "primary",               
            "LogLoss": 0.5996,
            "AUC": 0.7675,
            "F1": 0.6670,
            "Precision@10": 0.7653,
            "Recall@10": 0.9938,
            "NDCG@10": 0.9298,
            "HR@10": 0.9996,
            "MRR": 0.9154,
            "early_stopping": True,
            "saved": True
        },
        "LightGBM": {
            "status": "challenger",
            "AUC": 0.6036,
            "F1": 0.6372,
            "Precision@10": 0.1295,
            "MRR": 0.9232,
            "NDCG@10": 0.9414,
            "early_stopping": True,            
            "saved": False,                    
            "note": "Precision@10 degenerate — needs feature fix"
        },
        "CatBoost": {
            "status": "challenger",
            "AUC": 0.6044,
            "F1": 0.6312,
            "Precision@10": 0.1295,
            "MRR": 0.9297,
            "NDCG@10": 0.9463,
            "early_stopping": True,            
            "saved": False,
            "note": "Precision@10 degenerate — needs feature fix"
        },
        "LogisticRegression": {
            "status": "baseline_reference",
            "AUC": 0.5900,
            "F1": 0.5905,
            "Precision@10": 0.1295,
            "MRR": 0.9296,
            "NDCG@10": 0.9461,
            "early_stopping": False,           
            "saved": False
        },
        "RandomForest": {
            "status": "baseline_reference",
            "AUC": 0.5973,
            "F1": 0.6306,
            "Precision@10": 0.1295,
            "MRR": 0.9261,
            "NDCG@10": 0.9437,
            "early_stopping": False,           
            "saved": False
        },
        "NCF": {
            "status": "deprecated",
            "BCE": 1.4646,
            "AUC": 0.5689,
            "F1": 0.0,
            "Precision@10": 0.6785,
            "Recall@10": 0.9716,
            "NDCG@10": 0.8596,
            "HR@10": 0.9969,
            "MRR": 0.841,
            "early_stopping": False,
            "saved": False
        }
    },
    "promotion_thresholds": {           
        "AUC": 0.7675,
        "F1": 0.6670,
        "Precision@10": 0.7653
    }
}

collection.update_one(
    {"primary_model": "XGB"},
    {"$set": metrics_doc},
    upsert=True
)

print("MongoDB metrics updated with structured thresholds and status.")
