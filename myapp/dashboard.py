from flask import Blueprint, render_template, request, current_app, jsonify, flash, Response, send_file
from flask_login import login_required, current_user
from .extensions import db
from .models import PredictionRecord

import os, json, io, csv, datetime as dt
import numpy as np
import joblib

dash_bp = Blueprint("dash", __name__)

# cache model biar gak load berulang
_model_cache = {"clf": None, "reg": None, "meta_clf": {}, "meta_reg": {}}

# ---------- helpers umum ----------
def _load_meta(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
    
# helper: parse tanggal input (yyyy-mm-dd) -> date; default today
def _parse_start_date(s: str | None) -> dt.date:
    if not s:
        return dt.date.today()
    try:
        return dt.datetime.strptime(s.strip(), "%Y-%m-%d").date()
    except Exception:
        return dt.date.today()

def _last_estimator(pipe):
    if hasattr(pipe, "named_steps") and pipe.named_steps:
        return list(pipe.named_steps.values())[-1]
    return pipe

def _as_float(val):
    if val is None or val == "":
        return None
    try:
        return float(val)
    except Exception:
        return None

def _safe_feature_names_in(obj) -> list:
    val = getattr(obj, "feature_names_in_", None)
    if val is None:
        return []
    if hasattr(val, "tolist"):
        try:
            return list(val.tolist())
        except Exception:
            pass
    try:
        return list(val)
    except Exception:
        return []

# ---------- normalisasi kunci (fitur model → kolom DB) ----------
_DB_COLS = {c.name for c in PredictionRecord.__table__.columns}

def _to_db_key(k: str) -> str:
    """
    Ubah nama fitur yang mungkin mengandung satuan/simbol menjadi
    nama kolom DB versi “bersih”.
    contoh:
      suhu_udara_°c -> suhu_udara
      kelembaban_udara_% -> kelembapan_udara
    """
    k = (k or "").strip().lower()
    # ejaan
    k = k.replace("kelembaban", "kelembapan")
    # hapus suffix satuan yang sering muncul saat training
    for suf in ("_°c", "_%", "_mm", "_mg_kg"):
        k = k.replace(suf, "")
    return k

def _vals_for_db(vals_model: dict) -> dict:
    """
    Ambil dict {fitur_model: nilai} → {kolom_db: nilai}, buang yang tidak ada di tabel.
    """
    out = {}
    for k, v in (vals_model or {}).items():
        dk = _to_db_key(k)
        if dk in _DB_COLS:
            out[dk] = v
    return out

def _collect_vals(feats: list, src) -> dict:
    """
    Ambil nilai input untuk setiap fitur:
    - coba kunci original (mis. 'suhu_udara_°c')
    - jika kosong, coba versi DB ('suhu_udara')
    src harus punya .get (request.form / dict JSON).
    """
    out = {}
    for feat in feats:
        raw = src.get(feat)
        if raw in (None, ""):
            alt = _to_db_key(feat)
            raw = src.get(alt)
        out[feat] = _as_float(raw)
    return out

# ---------- rekomendasi & aturan hari ----------
def _build_rekomendasi(status_label: str, features_dict: dict) -> str:
    if status_label == "Kurang Subur":
        return "Tambah pupuk sesuai kekurangan (N, P, K) dan perbaiki pH"
    if status_label == "Sedang":
        return "Pantau dan sesuaikan pupuk jika perlu"
    if status_label == "Sangat Subur":
        return "Pertahankan kondisi saat ini"
    return "Periksa sensor pH dan NPK"

def _predict_status(pipe, X, meta) -> str:
    y_hat = pipe.predict(X)[0]
    if isinstance(y_hat, str):
        return y_hat
    classes = meta.get("classes") or []
    try:
        return classes[int(y_hat)]
    except Exception:
        return str(y_hat)

def _rule_waktu_tanam(status: str) -> int:
    s = (status or "").strip().lower()
    if s in ("sangat subur", "subur"):
        return 45
    if s == "sedang":
        return 90
    return 120

def _find_days_model_path() -> str | None:
    """
    Cari file model regressor dari beberapa kandidat:
    - dari config (MODEL_PATH_DAYS / WAKTU_MODEL_PATH)
    - default XGB
    - default RF
    """
    candidates = [
        current_app.config.get("MODEL_PATH_DAYS"),
        current_app.config.get("WAKTU_MODEL_PATH"),
        "models/waktu_tanam_xgb_reg.pkl",
        "models/waktu_tanam_rf_reg.pkl",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None

def _compute_waktu_tanam(status: str, X: np.ndarray) -> int:
    use_reg = bool(current_app.config.get("USE_DAYS_REGRESSOR", False))
    meta_path = current_app.config.get("METADATA_PATH_DAYS") or "models/waktu_tanam_metadata.json"

    if not use_reg:
        return _rule_waktu_tanam(status)

    if _model_cache["reg"] is None:
        mdl_path = _find_days_model_path()
        if mdl_path:
            try:
                _model_cache["reg"] = joblib.load(mdl_path)
            except ModuleNotFoundError as e:
                # xgboost belum terpasang saat unpickle
                if "xgboost" in str(e).lower():
                    raise ModuleNotFoundError(
                        "Model regressor membutuhkan paket 'xgboost'. "
                        "Tambahkan ke requirements & install: pip install xgboost"
                    ) from e
                raise
            _model_cache["meta_reg"] = _load_meta(meta_path)
        else:
            current_app.config["USE_DAYS_REGRESSOR"] = False
            return _rule_waktu_tanam(status)

    reg = _model_cache["reg"]
    try:
        days = float(reg.predict(X)[0])
        return int(max(1.0, min(365.0, round(days))))
    except Exception:
        return _rule_waktu_tanam(status)

# ---------- load classifier ----------
def get_status_model():
    if _model_cache["clf"] is None:
        mdl_path = current_app.config.get("MODEL_PATH", "models/status_rf_clf.pkl")
        meta_path = current_app.config.get("METADATA_PATH", "models/status_metadata.json")

        if not os.path.exists(mdl_path):
            raise FileNotFoundError(f"Model file not found: {mdl_path}")

        try:
            pipe = joblib.load(mdl_path)
        except ModuleNotFoundError as e:
            if "xgboost" in str(e).lower():
                raise ModuleNotFoundError(
                    "Model classifier membutuhkan paket 'xgboost'. "
                    "Tambahkan ke requirements & install: pip install xgboost"
                ) from e
            raise

        meta = _load_meta(meta_path)

        # pastikan classifier
        last_est = _last_estimator(pipe)
        is_classifier = (meta.get("model_kind") == "classifier") or hasattr(last_est, "classes_")
        if not is_classifier:
            raise TypeError(
                f"Model di '{mdl_path}' bukan classifier. Pastikan metadata 'model_kind'='classifier'."
            )

        # tentukan urutan fitur
        feats_meta  = meta.get("features")
        feats_model = _safe_feature_names_in(pipe)
        feats_cfg   = list(current_app.config.get("FEATURE_NAMES") or [])

        if   feats_meta: chosen = feats_meta
        elif feats_model: chosen = feats_model
        elif feats_cfg:   chosen = feats_cfg
        else:
            raise ValueError("Tidak bisa menentukan urutan fitur (metadata/pipe/config kosong).")

        if current_app.config.get("ALLOW_METADATA_FEATURES_OVERRIDE", True):
            current_app.config["FEATURE_NAMES"] = chosen

        _model_cache["clf"] = pipe
        _model_cache["meta_clf"] = meta

    return _model_cache["clf"], _model_cache["meta_clf"]

# =================== ROUTES ===================

@dash_bp.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    if request.method == "GET":
        return render_template("dashboard.html")

    feats = current_app.config["FEATURE_NAMES"]
    f = request.form

    lokasi = (f.get("lokasi_tanam") or "").strip() or None

    # ambil nilai versi fitur-model (boleh beda nama dengan form)
    vals_model = _collect_vals(feats, f)
    X = np.array([[vals_model.get(name, np.nan) for name in feats]], dtype=float)

    clf, meta = get_status_model()
    label = _predict_status(clf, X, meta)
    days  = _compute_waktu_tanam(label, X)
    rekom = _build_rekomendasi(label, vals_model)

    # konversi ke kolom DB
    vals_db = _vals_for_db(vals_model)

    start_date = dt.date.today()
    target_date = start_date + dt.timedelta(days=int(days))
    target_date_iso  = target_date.isoformat()           # untuk DB (YYYY-MM-DD)
    target_date_disp = target_date.strftime("%d/%m/%Y")  # untuk UI

    rec = PredictionRecord(
        user_id=current_user.id,
        lokasi_tanam=lokasi,
        status_kesuburan=label,
        rekomendasi=rekom,
        waktu_tanam_hari=int(days),
        waktu_tanam_tanggal=target_date_iso
        **vals_db
    )
    db.session.add(rec)
    db.session.commit()
    flash("Prediksi tersimpan.", "success")

    return render_template(
        "dashboard.html",
        result={
            "status_kesuburan": label,
            "rekomendasi": rekom,
            "waktu_tanam_hari": int(days),
            "waktu_tanam_tanggal": target_date_disp,
            "lokasi_tanam": lokasi,
        },
        inputs=vals_db
    )

@dash_bp.post("/api/predict")
@login_required
def api_predict():
    data = request.get_json(silent=True) or {}
    feats = current_app.config["FEATURE_NAMES"]

    lokasi = (data.get("lokasi_tanam") or "").strip() or None
    start_str = (data.get("tanggal_input") or "").strip() or None
    start_date = _parse_start_date(start_str)

    vals_model = _collect_vals(feats, data)
    X = np.array([[vals_model.get(name, np.nan) for name in feats]], dtype=float)

    clf, meta = get_status_model()
    label = _predict_status(clf, X, meta)
    days  = _compute_waktu_tanam(label, X)
    rekom = _build_rekomendasi(label, vals_model)

    target_date = start_date + dt.timedelta(days=int(days))
    target_date_iso  = target_date.isoformat()
    target_date_str  = target_date.strftime("%d/%m/%Y")

    vals_db = _vals_for_db(vals_model)

    rec = PredictionRecord(
        user_id=current_user.id,
        lokasi_tanam=lokasi,
        status_kesuburan=label,
        rekomendasi=rekom,
        waktu_tanam_hari=int(days),
        waktu_tanam_tanggal=target_date_iso,
        **vals_db
    )
    db.session.add(rec); db.session.commit()

    return jsonify(
        ok=True,
        id=rec.id,
        status_kesuburan=label,
        rekomendasi=rekom,
        waktu_tanam_hari=int(days),
        waktu_tanam_tanggal=target_date_str,  # << dikembalikan di API juga
        inputs=vals_db,
        lokasi_tanam=lokasi,
    )

@dash_bp.get("/debug/model")
@login_required
def debug_model():
    clf, meta = get_status_model()
    feats = current_app.config["FEATURE_NAMES"]
    last = _last_estimator(clf)

    reg_path = _find_days_model_path()
    reg_exists = bool(reg_path and os.path.exists(reg_path))
    return jsonify({
        "features_in_use": feats,
        "is_classifier": hasattr(last, "classes_") or (meta.get("model_kind") == "classifier"),
        "meta": meta,
        "use_days_regressor": bool(current_app.config.get("USE_DAYS_REGRESSOR", False)),
        "days_regressor_exists": reg_exists,
        "days_regressor_path": reg_path,
    })

def get_days_regressor():
    if not current_app.config.get("USE_DAYS_REGRESSOR", False):
        return None, {}
    if _model_cache["reg"] is None:
        mdl_path = _find_days_model_path()
        meta_path = current_app.config.get("METADATA_PATH_DAYS")
        if not mdl_path:
            current_app.config["USE_DAYS_REGRESSOR"] = False
            return None, {}
        try:
            pipe = joblib.load(mdl_path)
        except ModuleNotFoundError as e:
            if "xgboost" in str(e).lower():
                raise ModuleNotFoundError(
                    "Model regressor membutuhkan paket 'xgboost'. "
                    "Tambahkan ke requirements & install: pip install xgboost"
                ) from e
            raise
        meta = _load_meta(meta_path)
        _model_cache["reg"] = pipe
        _model_cache["meta_reg"] = meta
    return _model_cache["reg"], _model_cache["meta_reg"]

# ---------- LAYAR PREDIKSI LAMA ----------
@dash_bp.route("/prediksi", methods=["GET", "POST"])
@login_required
def prediksi():
    if request.method == "GET":
        today_str = dt.date.today().strftime("%Y-%m-%d")
        return render_template("prediksi.html", today_str=today_str)

    feats = current_app.config["FEATURE_NAMES"]
    f = request.form

    lokasi = (f.get("lokasi_tanam") or "").strip() or None
    # << ambil tanggal input dari form (opsional)
    start_str = (f.get("tanggal_input") or "").strip() or None
    start_date = _parse_start_date(start_str)

    # nilai-nilai fitur sesuai urutan model
    vals_model = _collect_vals(feats, f)
    X = np.array([[vals_model.get(name, np.nan) for name in feats]], dtype=float)

    clf, meta = get_status_model()
    label = _predict_status(clf, X, meta)
    rekom = _build_rekomendasi(label, vals_model)

    reg, _ = get_days_regressor()
    if reg is not None:
        try:
            days = float(reg.predict(X)[0])
            days = max(1.0, min(365.0, round(days)))
        except Exception:
            days = 120.0
    else:
        # fallback ke aturan Excel
        days = float(_rule_waktu_tanam(label))

    # === hitung tanggal target = start_date + days ===
    target_date = start_date + dt.timedelta(days=int(days))
    target_date_iso  = target_date.isoformat()
    target_date_str  = target_date.strftime("%d/%m/%Y")

    # simpan ke DB (tetap simpan integer harinya saja)
    vals_db = _vals_for_db(vals_model)
    rec = PredictionRecord(
        user_id=current_user.id,
        lokasi_tanam=lokasi,
        status_kesuburan=label,
        rekomendasi=rekom,
        waktu_tanam_hari=int(days),
        waktu_tanam_tanggal=target_date_iso,
        **vals_db
    )
    db.session.add(rec); db.session.commit()
    flash("Prediksi tersimpan.", "success")

    today_str = dt.date.today().strftime("%Y-%m-%d")
    return render_template(
        "prediksi.html",
        today_str=today_str,
        result={
            "status_kesuburan": label,
            "rekomendasi": rekom,
            "waktu_tanam_hari": int(days),
            "waktu_tanam_tanggal": target_date_str,   # << tampilkan di UI
            "lokasi_tanam": lokasi,
        },
        inputs=vals_db
    )

# ---------- Laporan & Export ----------
@dash_bp.route("/laporan", methods=["GET"])
@login_required
def laporan():
    rows = PredictionRecord.query.order_by(PredictionRecord.id.desc()).all()
    return render_template("laporan.html", rows=rows)

_EXPORT_COLS = [
    ("timestamp", "created_at"),
    ("lokasi_tanam", "lokasi_tanam"),
    ("suhu_udara", "suhu_udara"),
    ("kelembapan_udara", "kelembapan_udara"),
    ("suhu_tanah", "suhu_tanah"),
    ("kelembapan_tanah", "kelembapan_tanah"),
    ("ph_tanah", "ph_tanah"),
    ("nitrogen", "nitrogen"),
    ("fosfor", "fosfor"),
    ("kalium", "kalium"),
    ("curah_hujan", "curah_hujan"),
    ("status_kesuburan", "status_kesuburan"),
    ("rekomendasi", "rekomendasi"),
    ("waktu_tanam_hari", "waktu_tanam_hari"),
    ("waktu_tanam_tanggal", "waktu_tanam_tanggal"),
]

def _records_to_rows(queryset):
    rows = []
    for r in queryset:
        d = {}
        for label, attr in _EXPORT_COLS:
            val = getattr(r, attr, None)
            if isinstance(val, dt.datetime):
                val = val.strftime("%Y-%m-%d %H:%M:%S")
            d[label] = val
        rows.append(d)
    return rows

@dash_bp.get("/laporan/export.csv")
@login_required
def export_csv():
    q = PredictionRecord.query.order_by(PredictionRecord.id.desc()).all()
    rows = _records_to_rows(q)
    si = io.StringIO()
    writer = csv.DictWriter(si, fieldnames=[c[0] for c in _EXPORT_COLS])
    writer.writeheader()
    writer.writerows(rows)
    out = si.getvalue().encode("utf-8-sig")
    return Response(out, mimetype="text/csv",
                    headers={"Content-Disposition":"attachment; filename=laporan_eucagrow.csv"})

@dash_bp.get("/laporan/export.xlsx")
@login_required
def export_xlsx():
    try:
        import pandas as pd
    except Exception:
        flash("Paket pandas/openpyxl belum terinstal. Tambahkan ke requirements.", "error")
        return render_template("laporan.html", rows=PredictionRecord.query.order_by(PredictionRecord.id.desc()).all())

    q = PredictionRecord.query.order_by(PredictionRecord.id.desc()).all()
    rows = _records_to_rows(q)
    df = pd.DataFrame(rows, columns=[c[0] for c in _EXPORT_COLS])
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Laporan", index=False)
    bio.seek(0)
    return send_file(bio, as_attachment=True,
                     download_name="laporan_eucagrow.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ==== PDF helpers ============================================================
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics

_NUM_COLS = [3,4,5,6,7,8,9,10,13]

def _fmt_cell(v):
    if v is None: return "-"
    try:
        if isinstance(v, (int, float)):
            return f"{v:.1f}"
    except Exception:
        pass
    return str(v)

def _build_pdf_rows(dict_rows):
    headers = [
        "Waktu","Lokasi","Suhu Udara","Kelemb. Udara","Suhu Tanah","Kelemb. Tanah",
        "pH","N","P","K","Curah Hujan","Status","Rekomendasi","Waktu (hari)"
    ]
    keys = [
        "timestamp","lokasi_tanam","suhu_udara","kelembapan_udara","suhu_tanah",
        "kelembapan_tanah","ph_tanah","nitrogen","fosfor","kalium","curah_hujan",
        "status_kesuburan","rekomendasi","waktu_tanam_hari"
    ]
    cell = ParagraphStyle("cell", fontName="Helvetica", fontSize=8, leading=9.6, spaceBefore=0, spaceAfter=0, wordWrap="CJK")
    data = [headers]
    for d in dict_rows:
        row = []
        for k in keys:
            txt = _fmt_cell(d.get(k))
            if k in ("rekomendasi","lokasi_tanam","status_kesuburan"):
                row.append(Paragraph(txt, cell))
            else:
                row.append(txt)
        data.append(row)
    return data

def _auto_col_widths(data, avail_width):
    n = len(data[0])
    font, size = "Helvetica", 8
    pad = 12
    widths = [0]*n
    for row in data[: min(80, len(data))]:
        for i, cell in enumerate(row):
            txt = cell.getPlainText() if isinstance(cell, Paragraph) else str(cell)
            w = pdfmetrics.stringWidth(txt, font, size) + pad
            if w > widths[i]: widths[i] = w
    mins = [60, 80, 58, 72, 58, 72, 36, 36, 36, 36, 60, 64, 180, 70]
    maxs = [90, 150, 75, 90, 75, 90, 42, 45, 45, 45, 80, 100, 260, 90]
    widths = [max(mins[i], min(widths[i], maxs[i])) for i in range(n)]
    total = sum(widths)
    if total > avail_width:
        scale = avail_width / total
        widths = [w*scale for w in widths]
    return widths

def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillGray(0.4)
    canvas.drawRightString(doc.pagesize[0]-18, 12, f"Hal. {doc.page}")
    canvas.restoreState()

@dash_bp.get("/laporan/export.pdf")
@login_required
def export_pdf():
    try:
        from reportlab.lib.pagesizes import A4, landscape
    except Exception:
        flash("Paket reportlab belum terinstal. Tambahkan ke requirements.", "error")
        return render_template(
            "laporan.html",
            rows=PredictionRecord.query.order_by(PredictionRecord.id.desc()).all()
        )

    q = PredictionRecord.query.order_by(PredictionRecord.id.desc()).all()
    rows = _records_to_rows(q)
    data = _build_pdf_rows(rows)

    page_size = landscape(A4)
    left, right, top, bottom = 18, 18, 22, 18
    bio = io.BytesIO()
    doc = SimpleDocTemplate(
        bio, pagesize=page_size,
        leftMargin=left, rightMargin=right, topMargin=top, bottomMargin=bottom
    )

    title = Paragraph(
        "<b>Laporan Prediksi EUCAGROW</b>",
        ParagraphStyle("h", alignment=1, fontName="Helvetica-Bold", fontSize=12, leading=14)
    )
    sub = Paragraph(
        f"Digenerasi: {dt.datetime.now():%Y-%m-%d %H:%M:%S} • Total: {len(rows)} baris",
        ParagraphStyle("s", alignment=1, fontName="Helvetica", fontSize=8, textColor=colors.HexColor('#64748b'))
    )

    avail_width = page_size[0] - left - right
    col_widths = _auto_col_widths(data, avail_width)

    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    style = TableStyle([
        ("FONT", (0,0), (-1,-1), "Helvetica", 8),
        ("LEADING", (0,0), (-1,-1), 9.6),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F1F5F9")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#0F172A")),
        ("ALIGN", (0,0), (-1,0), "CENTER"),
        ("LINEBELOW", (0,0), (-1,0), 0.6, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#FAFAFA")]),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#E5E7EB")),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ])
    for c in _NUM_COLS:
        style.add("ALIGN", (c,1), (c,-1), "RIGHT")
    tbl.setStyle(style)

    elements = [title, sub, Spacer(0, 8), tbl]
    doc.build(elements, onFirstPage=_footer, onLaterPages=_footer)

    bio.seek(0)
    return send_file(
        bio, as_attachment=True,
        download_name="laporan_eucagrow.pdf",
        mimetype="application/pdf"
    )
