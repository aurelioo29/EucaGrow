# myapp/__init__.py
import os
from flask import Flask, request

from .extensions import db, login_manager, bcrypt


def create_app(config_object="config.Config"):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config_object)

    # Optional: reload template saat file berubah (berguna saat dev)
    app.config.setdefault("TEMPLATES_AUTO_RELOAD", True)

    # Init extensions
    db.init_app(app)
    login_manager.init_app(app)
    bcrypt.init_app(app)
    login_manager.login_view = "auth.login"

    # --- CACHE BUSTING UNTUK FILE STATIK (CSS/JS/IMG) ---
    # Pakai mtime file sebagai versi, dipakai di template: ?v={{ static_hash('css/base.css') }}
    @app.template_global()
    def static_hash(path: str) -> str:
        """Return last modified time of static file as cache-buster string."""
        full = os.path.join(app.static_folder, path)
        try:
            return str(int(os.path.getmtime(full)))
        except OSError:
            return "0"

    # --- NO-STORE UNTUK HALAMAN DINAMIS ---
    # Supaya HTML dinamis tidak di-cache oleh browser/back-forward cache.
    @app.after_request
    def add_cache_headers(resp):
        ep = (request.endpoint or "")
        # Jangan sentuh /static
        if ep.startswith("static"):
            # Jika kamu menambahkan ?v=... di URL, biarkan asset di-cache lama
            # Browser akan fetch ulang saat versinya berubah.
            return resp

        # Untuk semua response dinamis, cegah cache
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp

    # Register blueprints
    from .auth import auth_bp
    from .dashboard import dash_bp
    from .main import main_bp
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(dash_bp)

    # Buat tabel jika belum ada
    with app.app_context():
        db.create_all()

    return app
