from datetime import datetime
from flask_login import UserMixin
from .extensions import db, login_manager

class User(db.Model, UserMixin):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)  # ← username
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<User {self.username}>"

@login_manager.user_loader
def load_user(user_id: str):
    return User.query.get(int(user_id))

class PredictionRecord(db.Model):
    __tablename__ = "prediction_records"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    lokasi_tanam = db.Column(db.String(128), nullable=True)

    suhu_udara = db.Column(db.Float, nullable=True)
    kelembapan_udara = db.Column(db.Float, nullable=True)
    suhu_tanah = db.Column(db.Float, nullable=True)
    kelembapan_tanah = db.Column(db.Float, nullable=True)
    ph_tanah = db.Column(db.Float, nullable=True)
    nitrogen = db.Column(db.Float, nullable=True)
    fosfor = db.Column(db.Float, nullable=True)
    kalium = db.Column(db.Float, nullable=True)
    curah_hujan = db.Column(db.Float, nullable=True)

    status_kesuburan = db.Column(db.String(32), nullable=True)
    rekomendasi = db.Column(db.String(255), nullable=True)
    waktu_tanam_hari = db.Column(db.Integer, nullable=True)  # kamu minta fixed 120

    user = db.relationship("User", backref="predictions", lazy=True)