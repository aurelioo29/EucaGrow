import os
from dotenv import load_dotenv

# Load from instance/.env
load_dotenv(os.path.join("instance", ".env"))

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")

    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL",
        "mysql+pymysql://root:password@127.0.0.1:3306/flask_login_db"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ===== Model status (classifier) =====
    MODEL_PATH = os.getenv("MODEL_PATH", "models/status_xgb_clf.pkl")
    METADATA_PATH = os.getenv("METADATA_PATH", "models/status_metadata.json")

    # ===== (Opsional) Model hari tanam (regressor) =====
    USE_DAYS_REGRESSOR = os.getenv("USE_DAYS_REGRESSOR", "false").lower() == "true"
    MODEL_PATH_DAYS = os.getenv("MODEL_PATH_DAYS", "models/waktu_tanam_xgb_reg.pkl")
    METADATA_PATH_DAYS = os.getenv("METADATA_PATH_DAYS", "models/waktu_tanam_metadata.json")
    WAKTU_MODEL_PATH = os.getenv("WAKTU_MODEL_PATH", "models/waktu_tanam_xgb_reg.pkl")

    # Urutan fitur default (boleh dioverride oleh metadata saat runtime)
    FEATURE_NAMES = [
        "suhu_udara", "kelembapan_udara", "suhu_tanah", "kelembapan_tanah",
        "ph_tanah", "nitrogen", "fosfor", "kalium", "curah_hujan"
    ]

    # izinkan metadata menimpa FEATURE_NAMES
    ALLOW_METADATA_FEATURES_OVERRIDE = True
