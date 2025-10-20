# 🌵 Eucagrow 🌵

Flask-based web app for **vertical-garden management and prediction** — complete with auth, dashboard, reports, and an ML-driven _prediksi_ module. Built to be practical, fast to ship, and easy to extend.

## ✨ Features

- **Auth & Sessions** — login flow with session handling.
- **Dashboard** — quick stats and entry points to core modules.
- **Prediksi** — model inference endpoint + UI for plant/health predictions.
- **Laporan** — exportable reports view.
- **Static assets & clean templates** — split CSS per page.
- **Notebooks** — reproducible model training/evaluation steps.
- **Config via `.env`** — one place to tweak secrets and paths.

## 🧱 Tech Stack

- **Backend:** Flask (Python)
- **Frontend:** HTML, CSS (custom + page-specific), vanilla JS, Bootstrap (optional)
- **Data/ML:** Jupyter Notebooks (training), serialized model artifacts in `/models`
- **Runtime:** Gunicorn (prod), Flask dev server (local)

## 📦 Project Structure

```
eucagrow/
├── app.py
├── config.py
├── requirements.txt
├── requirements-prod.txt
├── instance/
│   └── .env
├── models/
├── myapp/
│   ├── __init__.py
│   ├── auth.py
│   ├── dashboard.py
│   ├── extensions.py
│   ├── main.py
│   ├── models.py
│   ├── static/
│   │   ├── css/
│   │   │   ├── base.css
│   │   │   ├── dashboard.css
│   │   │   ├── laporan.css
│   │   │   ├── login.css
│   │   │   └── prediksi.css
│   │   ├── bg.jpg
│   │   ├── pohon.webp
│   │   └── tanaman.webp
│   └── templates/
│       ├── base.html
│       ├── dashboard.html
│       ├── laporan.html
│       ├── login.html
│       └── prediksi.html
├── notebooks/               # Jupyter notebooks for experiments/training
└── .gitignore
```

## 🚀 Quickstart (Local)

### 1) Clone & create venv

```bash
git clone https://github.com/aurelioo29/EucaGrow.git
cd EucaGrow

# Windows (PowerShell)
python -m venv env
env\Scripts\activate

# macOS/Linux
python3 -m venv env && source env/bin/activate
```

### 2) Install deps

```bash
pip install -r requirements.txt
```

### 3) Environment

Copy the example file and edit values as needed:

```bash
# Windows
copy .env.example instance\.env

# macOS/Linux
# mkdir -p instance && cp .env.example instance/.env
```

### 4) Run

```bash
python app.py
```

App defaults to: `http://127.0.0.1:5000`

## ⚙️ Configuration

Key env vars (see `.env.example`):

- `SECRET_KEY` — Flask session key.
- `DATABASE_URL` — SQLAlchemy DB URI (defaults to MySQL).
- `MODEL_PATH` — path to your serialized model (e.g., `models/model.pkl`).
- `ENV` / `FLASK_ENV` — development or production.

## 🧪 Notebooks & Models

- Train or evaluate models in `/notebooks`.
- Save final artifacts into `/models` and point `MODEL_PATH` there.
- Inference is handled by the **prediksi** route/UI.

## 📤 Production (Gunicorn production)

```bash
pip install -r requirements-prod.txt

# If you expose a global 'app' in app.py:
gunicorn -w 4 -b 0.0.0.0:8000 "app:app"

# If using an app factory in myapp/__init__.py:
# gunicorn -w 4 -b 0.0.0.0:8000 "myapp:create_app()"
```

Behind Nginx, proxy to `127.0.0.1:8000`.

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request. ✨
