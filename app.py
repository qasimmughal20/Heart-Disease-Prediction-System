"""
Heart Disease Risk Prediction — Professional Web App
Full feature parity with GUI_App.py: PDF report, demo samples, history, admin dashboard, chart.
"""

import os, io, csv, datetime as dt
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

# ── Config ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CardioScan AI", page_icon="🫀", layout="wide")

FEATURES = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
DISPLAY  = {
    "age":"Age","sex":"Sex","cp":"Chest Pain Type","trestbps":"Resting Blood Pressure (mmHg)",
    "chol":"Cholesterol (mg/dl)","fbs":"Fasting Blood Sugar > 120 mg/dl","restecg":"Resting ECG",
    "thalach":"Max Heart Rate","exang":"Exercise Angina","oldpeak":"ST Depression (oldpeak)",
    "slope":"ST Slope","ca":"Major Vessels (0-4)","thal":"Thalassemia",
}
RANGES = {
    "age":(1,120),"sex":(0,1),"cp":(0,3),"trestbps":(70,250),"chol":(80,700),
    "fbs":(0,1),"restecg":(0,2),"thalach":(60,250),"exang":(0,1),
    "oldpeak":(0.0,10.0),"slope":(0,2),"ca":(0,4),"thal":(0,3),
}
NORMAL_REF = {"trestbps":120,"chol":200,"thalach":170,"oldpeak":1.0}
PATIENT_FILE = "heart_patient_prediction_records.csv"
ADMIN_USER, ADMIN_PASS = "admin", "admin123"

SAMPLES = {
    "🔴 High Risk — Ali Khan": {
        "name":"Ali Khan","contact":"03001234567",
        "age":63,"sex":1,"cp":0,"trestbps":180,"chol":320,"fbs":1,"restecg":2,
        "thalach":95,"exang":1,"oldpeak":6.0,"slope":2,"ca":4,"thal":3,
    },
    "🟡 Medium Risk — Sara Ahmed": {
        "name":"Sara Ahmed","contact":"03111234567",
        "age":48,"sex":0,"cp":1,"trestbps":140,"chol":220,"fbs":0,"restecg":1,
        "thalach":150,"exang":0,"oldpeak":1.8,"slope":1,"ca":1,"thal":2,
    },
    "🟢 Low Risk — Ahmed Raza": {
        "name":"Ahmed Raza","contact":"03221234567",
        "age":29,"sex":1,"cp":3,"trestbps":110,"chol":160,"fbs":0,"restecg":0,
        "thalach":190,"exang":0,"oldpeak":0.0,"slope":0,"ca":0,"thal":1,
    },
}

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=DM+Serif+Display&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #f0f4ff !important;
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f172a 0%, #1e3a5f 100%) !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { color: #cbd5e1 !important; font-size: 0.95rem; }
[data-testid="stSidebar"] .stRadio [data-checked="true"] label { color: #60a5fa !important; font-weight: 600; }

/* Fix: all form field labels clearly visible in main content */
[data-testid="stMain"] label,
[data-testid="stMain"] .stTextInput label,
[data-testid="stMain"] .stNumberInput label,
[data-testid="stMain"] .stSelectbox label,
[data-testid="stMain"] p {
    color: #0f172a !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}
/* White inputs with dark text and clear border */
[data-testid="stMain"] input {
    background: #ffffff !important;
    color: #0f172a !important;
}
[data-testid="stMain"] [data-baseweb="select"] {
    background: #ffffff !important;
    border: 1.5px solid #c7d2fe !important;
    border-radius: 8px !important;
}
[data-testid="stMain"] [data-baseweb="select"] * { color: #0f172a !important; background:#fff !important; }
[data-testid="stMain"] [data-baseweb="input"] { background:#ffffff !important; border:1.5px solid #c7d2fe !important; border-radius:8px !important; }

.hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e40af 60%, #0ea5e9 100%);
    border-radius: 18px; padding: 38px 44px; margin-bottom: 28px;
    display: flex; align-items: center; gap: 28px;
    box-shadow: 0 8px 40px rgba(15,23,42,0.18);
}
.hero-icon { font-size: 3.8rem; line-height:1; }
.hero-title { font-family:'DM Serif Display',serif; font-size:2.4rem; color:#fff; margin:0; letter-spacing:-0.5px; }
.hero-sub { color:#93c5fd; font-size:1.05rem; margin-top:6px; font-weight:400; }

.metric-row { display:flex; gap:14px; margin-bottom:24px; flex-wrap:wrap; }
.metric-card {
    flex:1; min-width:130px; background:#fff; border-radius:14px; padding:20px 18px;
    box-shadow:0 2px 12px rgba(37,99,235,0.08); border-top:4px solid #2563eb; text-align:center;
}
.metric-val { font-size:1.9rem; font-weight:800; color:#1e40af; }
.metric-lbl { font-size:0.78rem; color:#64748b; font-weight:600; letter-spacing:.04em; text-transform:uppercase; margin-top:2px; }

.section-card {
    background:#fff; border-radius:16px; padding:28px 28px 22px;
    box-shadow:0 2px 16px rgba(37,99,235,0.07); margin-bottom:22px;
    border:1px solid #e8f0fe;
}
.section-title { font-size:1.1rem; font-weight:700; color:#0f172a; margin-bottom:16px; display:flex; align-items:center; gap:8px; }

.result-high   { background:linear-gradient(135deg,#fef2f2,#fee2e2); border:2px solid #fca5a5; border-radius:16px; padding:28px; text-align:center; }
.result-medium { background:linear-gradient(135deg,#fffbeb,#fef3c7); border:2px solid #fcd34d; border-radius:16px; padding:28px; text-align:center; }
.result-low    { background:linear-gradient(135deg,#f0fdf4,#dcfce7); border:2px solid #86efac; border-radius:16px; padding:28px; text-align:center; }
.result-none   { background:#f8fafc; border:2px dashed #cbd5e1; border-radius:16px; padding:28px; text-align:center; }

.result-label  { font-size:1.05rem; font-weight:600; margin-bottom:6px; }
.result-pred   { font-size:1.35rem; font-weight:800; margin-bottom:4px; }
.result-prob   { font-size:2.6rem; font-weight:900; letter-spacing:-1px; }
.result-level  { font-size:1.15rem; font-weight:700; margin-top:4px; }
.result-expl   { font-size:0.88rem; color:#475569; margin-top:12px; line-height:1.5; }

.rec-item { background:#f1f5f9; border-left:4px solid #2563eb; border-radius:0 8px 8px 0;
            padding:10px 14px; margin-bottom:8px; font-size:0.93rem; color:#1e293b; }

.chart-wrap { background:#f8faff; border-radius:12px; padding:18px 14px 10px; }
.bar-row { display:flex; align-items:flex-end; gap:6px; margin-bottom:16px; }
.bar-group { display:flex; flex-direction:column; align-items:center; gap:4px; flex:1; }
.bar-pair { display:flex; align-items:flex-end; gap:3px; height:110px; }
.bar { border-radius:4px 4px 0 0; min-width:22px; transition:height .3s; }
.bar-p { background:#2563eb; }
.bar-n { background:#16a34a; }
.bar-lbl { font-size:0.72rem; font-weight:700; color:#475569; }
.bar-val { font-size:0.68rem; color:#64748b; }
.chart-legend { display:flex; gap:20px; justify-content:center; margin-top:8px; }
.leg-dot { width:12px; height:12px; border-radius:3px; display:inline-block; margin-right:5px; vertical-align:middle; }

.guide-item { display:flex; gap:10px; padding:7px 0; border-bottom:1px solid #f1f5f9; font-size:0.875rem; }
.guide-key { font-weight:600; color:#1e40af; min-width:130px; }
.guide-val { color:#475569; }

.stButton>button {
    background: linear-gradient(135deg,#1d4ed8,#2563eb) !important;
    color:#fff !important; border:none !important; border-radius:10px !important;
    font-weight:700 !important; font-size:1rem !important; padding:10px 24px !important;
    box-shadow:0 4px 14px rgba(37,99,235,0.3) !important; transition:all .2s !important;
    width:100% !important;
}
.stButton>button:hover { background:linear-gradient(135deg,#1e40af,#1d4ed8) !important; transform:translateY(-1px) !important; }
.btn-clear>button { background:linear-gradient(135deg,#475569,#64748b) !important; color:#fff !important; }
.btn-pdf>button   { background:linear-gradient(135deg,#059669,#10b981) !important; color:#fff !important; }
.btn-high>button  { background:linear-gradient(135deg,#dc2626,#ef4444) !important; color:#fff !important; }
.btn-mid>button   { background:linear-gradient(135deg,#d97706,#f59e0b) !important; color:#fff !important; }
.btn-low>button   { background:linear-gradient(135deg,#16a34a,#22c55e) !important; color:#fff !important; }

/* Download buttons (stDownloadButton) — always white text, green background */
[data-testid="stDownloadButton"]>button {
    background: linear-gradient(135deg,#059669,#10b981) !important;
    color:#fff !important; border:none !important; border-radius:10px !important;
    font-weight:700 !important; font-size:0.95rem !important; padding:10px 20px !important;
    box-shadow:0 4px 14px rgba(5,150,105,0.3) !important; width:100% !important;
    transition:all .2s !important;
}
[data-testid="stDownloadButton"]>button:hover {
    background: linear-gradient(135deg,#047857,#059669) !important;
    color:#fff !important; transform:translateY(-1px) !important;
}
[data-testid="stDownloadButton"]>button * { color:#fff !important; }

.admin-stat { background:#fff; border-radius:12px; padding:16px 20px; text-align:center;
              box-shadow:0 2px 10px rgba(0,0,0,0.06); border-top:3px solid #7c3aed; }
.admin-val  { font-size:2rem; font-weight:800; color:#7c3aed; }
.admin-lbl  { font-size:0.78rem; color:#64748b; font-weight:600; }

.disclaimer {
    background:#fff7ed; border:1px solid #fed7aa; border-radius:10px;
    padding:12px 18px; font-size:0.82rem; color:#92400e; margin-top:14px;
}
</style>
""", unsafe_allow_html=True)

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model on your dataset…")
def train_model():
    df = pd.read_csv("heart.csv")
    cols = {c.lower().replace(" ","").replace("_",""): c for c in df.columns}
    rename = {}
    for f in FEATURES:
        k = f.lower().replace("_","")
        if k in cols: rename[cols[k]] = f
    df = df.rename(columns=rename)
    clean = df[FEATURES+["target"]].copy()
    for c in FEATURES+["target"]:
        clean[c] = pd.to_numeric(clean[c], errors="coerce")
    clean = clean.dropna(subset=["target"])
    clean["target"] = (clean["target"].astype(float)>0).astype(int)
    clean = clean.drop_duplicates(subset=FEATURES+["target"]).reset_index(drop=True)
    # Note: label flip removed — new merged heart.csv has correct labels (1=disease, 0=no disease)
    X, y = clean[FEATURES], clean["target"]
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=0.7, solver="liblinear")),
    ])
    n_splits = max(2, min(5, int(y.value_counts().min())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    sc = cross_validate(pipe, X, y, cv=cv,
                        scoring={"acc":"accuracy","prec":"precision","rec":"recall","f1":"f1"})
    metrics = {k: round(float(sc[f"test_{k}"].mean())*100,1)
               for k in ["acc","prec","rec","f1"]}
    pipe.fit(X, y)
    return pipe, metrics, len(clean)

def predict(model, values):
    X = pd.DataFrame([values], columns=FEATURES)
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])
    level = "High Risk" if prob>=0.70 else "Medium Risk" if prob>=0.40 else "Low Risk"
    return pred, prob, level

# ── Helpers ────────────────────────────────────────────────────────────────────
def recommendations(values, prob):
    r = []
    if prob>=0.40: r.append("Consult a cardiologist or qualified physician as soon as possible.")
    else: r.append("Maintain routine checkups and preventive care.")
    if values["trestbps"]>=140: r.append("Reduce salt intake and monitor blood pressure regularly.")
    if values["chol"]>=240:     r.append("Avoid oily/fried foods and reduce saturated fat intake.")
    if values["thalach"]<120:   r.append("Discuss low heart-rate response with a medical professional.")
    if values["exang"]==1:      r.append("Avoid heavy exertion until reviewed by a doctor.")
    r.append("Exercise 30 minutes daily if approved by your doctor.")
    r.append("Schedule a medical checkup and keep records of BP, cholesterol, and symptoms.")
    return r

def explanation(values):
    reasons = []
    if values["trestbps"]>=140: reasons.append("high resting blood pressure")
    if values["chol"]>=240:     reasons.append("high cholesterol")
    if values["exang"]==1:      reasons.append("exercise-induced angina")
    if values["oldpeak"]>=2:    reasons.append("elevated ST depression")
    if values["ca"]>=2:         reasons.append("multiple major vessels affected")
    if not reasons: reasons.append("values close to healthy reference ranges")
    return "Result influenced by: " + ", ".join(reasons) + ". This is a screening result, not a final diagnosis."

def save_record(name, contact, values, pred_text, prob, level):
    row = {"Date": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
           "Patient Name": name, "Contact": contact,
           **values, "Prediction": pred_text,
           "Risk Probability %": round(prob*100,2), "Risk Level": level}
    exists = os.path.exists(PATIENT_FILE)
    with open(PATIENT_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists: w.writeheader()
        w.writerow(row)

def make_pdf(report):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    title_s = ParagraphStyle("T", parent=styles["Title"],
                              textColor=rl_colors.HexColor("#dc2626"), fontSize=22, leading=28)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"],
                         textColor=rl_colors.HexColor("#1d4ed8"), spaceAfter=6)
    r = report
    story = [
        Paragraph("❤ Heart Disease Risk Prediction Report", title_s), Spacer(1,8),
        Paragraph("<b>CardioScan AI — Educational Screening System</b>", styles["Normal"]),
        Paragraph(f"<b>Report Date:</b> {r['date']}", styles["Normal"]), Spacer(1,12),
    ]
    summary = Table(
        [["Patient Name", r["name"]], ["Contact Number", r["contact"]],
         ["Prediction", r["pred_text"]], ["Risk Level", r["level"]],
         ["Risk Probability", f"{r['prob']*100:.2f}%"], ["Alert", r["alert"]]],
        colWidths=[160, 320]
    )
    summary.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(0,-1),rl_colors.HexColor("#dbeafe")),
        ("GRID",(0,0),(-1,-1),.5,rl_colors.grey),
        ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
    ]))
    story += [summary, Spacer(1,14)]
    attrs = Table(
        [["Medical Attribute","Patient Value"]] + [[DISPLAY[k], str(v)] for k,v in r["values"].items()],
        hAlign="LEFT", colWidths=[270,150]
    )
    attrs.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),rl_colors.HexColor("#1d4ed8")),
        ("TEXTCOLOR",(0,0),(-1,0),rl_colors.white),
        ("GRID",(0,0),(-1,-1),.5,rl_colors.grey),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[rl_colors.white, rl_colors.HexColor("#f0f4ff")]),
    ]))
    story += [Paragraph("Patient Clinical Data", h2), attrs, Spacer(1,14)]
    story += [Paragraph("Clinical Explanation", h2),
              Paragraph(r["expl"], styles["Normal"]), Spacer(1,10)]
    story += [Paragraph("Lifestyle & Medical Recommendations", h2)]
    for rec in r["recs"]:
        story.append(Paragraph("• " + rec, styles["Normal"]))
    story += [
        Spacer(1,20),
        Paragraph("<b>Reviewing Doctor:</b> ____________________________", styles["Normal"]),
        Spacer(1,8),
        Paragraph("<b>Date:</b> ____________________________  <b>Signature:</b> ____________________________", styles["Normal"]),
        Spacer(1,14),
        Paragraph("<i>Disclaimer: This is an educational ML screening report generated by CardioScan AI. "
                  "It is not a final medical diagnosis. Always consult a qualified medical professional.</i>",
                  styles["Italic"]),
    ]
    doc.build(story)
    return buf.getvalue()

# ── Session State ──────────────────────────────────────────────────────────────
for k, v in {"last_report": None, "admin_logged_in": False, "page": "Prediction",
              "demo_loaded": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:24px 0 18px;'>
      <div style='font-size:2.6rem;'>🫀</div>
      <div style='font-family:"DM Serif Display",serif;font-size:1.5rem;color:#fff;margin-top:6px;'>CardioScan AI</div>
      <div style='color:#94a3b8;font-size:0.82rem;margin-top:4px;'>Heart Disease Risk Predictor</div>
    </div>
    <hr style='border-color:#334155;margin:0 0 18px;'>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", ["🏥 Prediction", "📋 Patient History", "🔐 Admin Dashboard"],
                    label_visibility="collapsed")
    st.session_state.page = page

    st.markdown("<hr style='border-color:#334155;margin:20px 0 14px;'>", unsafe_allow_html=True)
    model, metrics, n_rows = train_model()
    st.markdown(f"""
    <div style='font-size:0.78rem;color:#94a3b8;text-align:center;line-height:1.8;'>
      Model: Logistic Regression<br>
      Dataset: {n_rows} unique patients<br>
      Sources: Cleveland · Hungary · Switzerland · VA Long Beach · Statlog<br>
      CV Accuracy: <b style='color:#60a5fa;'>{metrics['acc']}%</b><br>
      5-Fold Stratified CV
    </div>
    <div style='margin-top:18px;font-size:0.72rem;color:#64748b;text-align:center;line-height:1.6;padding:0 8px;'>
      Educational screening tool only.<br>Not a medical device.<br>Consult a doctor for diagnosis.
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "🏥 Prediction":

    st.markdown("""
    <div class='hero-banner'>
      <div class='hero-icon'>🫀</div>
      <div>
        <div class='hero-title'>CardioScan AI</div>
        <div class='hero-sub'>Heart Disease Risk Prediction System &nbsp;·&nbsp; Powered by Machine Learning</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Model metrics moved to Admin Dashboard

    left, right = st.columns([1.55, 1], gap="large")

    with left:
        demo = st.session_state.demo_loaded or {}

        # ── Patient Info ─────────────────────────────────────────────────────
        st.markdown("<p style='font-size:1.1rem;font-weight:700;color:#0f172a;margin:0 0 8px;'>👤 Patient Information</p>", unsafe_allow_html=True)
        st.divider()
        pc1, pc2 = st.columns(2)
        p_name    = pc1.text_input("Patient Name *", value=demo.get("name",""), placeholder="Enter full name")
        p_contact = pc2.text_input("Contact Number *", value=demo.get("contact",""), placeholder="e.g. 03001234567")

        # ── Clinical Data ─────────────────────────────────────────────────────
        st.markdown("<p style='font-size:1.1rem;font-weight:700;color:#0f172a;margin:18px 0 8px;'>🩺 Clinical Data</p>", unsafe_allow_html=True)
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            age      = st.number_input("Age (years)", 1, 120, int(demo.get("age", 1)))
            sex      = st.selectbox("Sex", [1, 0],
                                    index=0 if demo.get("sex", 1) == 1 else 1,
                                    format_func=lambda x: "Male" if x else "Female")
            # Chest Pain Type — null by default, user must choose
            _cp_opts = [None, 0, 1, 2, 3]
            _cp_default = demo.get("cp", None)
            _cp_idx = _cp_opts.index(_cp_default) if _cp_default in _cp_opts else 0
            cp = st.selectbox("Chest Pain Type *", _cp_opts,
                              index=_cp_idx,
                              format_func=lambda x: "— Select —" if x is None else
                              ["Typical Angina","Atypical Angina","Non-anginal Pain","Asymptomatic"][x])
            trestbps = st.number_input("Resting BP (mmHg)",    70,  250, int(demo.get("trestbps", 70)))
            chol     = st.number_input("Cholesterol (mg/dl)",  80,  700, int(demo.get("chol", 80)))
            fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1],
                                    index=int(demo.get("fbs", 0)),
                                    format_func=lambda x: "Yes" if x else "No")
            restecg  = st.selectbox("Resting ECG", [0, 1, 2],
                                    index=int(demo.get("restecg", 0)),
                                    format_func=lambda x: ["Normal","ST-T Abnormality","LV Hypertrophy"][x])
        with c2:
            thalach  = st.number_input("Max Heart Rate (bpm)",  60,  250, int(demo.get("thalach", 60)))
            exang    = st.selectbox("Exercise-Induced Angina", [0, 1],
                                    index=int(demo.get("exang", 0)),
                                    format_func=lambda x: "Yes" if x else "No")
            oldpeak  = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, float(demo.get("oldpeak", 0.0)), step=0.1)
            # ST Slope — null by default, user must choose
            _sl_opts = [None, 0, 1, 2]
            _sl_default = demo.get("slope", None)
            _sl_idx = _sl_opts.index(_sl_default) if _sl_default in _sl_opts else 0
            slope = st.selectbox("ST Slope *", _sl_opts,
                                 index=_sl_idx,
                                 format_func=lambda x: "— Select —" if x is None else
                                 ["Upsloping","Flat","Downsloping"][x])
            ca       = st.selectbox("Major Vessels (0–4)", [0, 1, 2, 3, 4],
                                    index=int(demo.get("ca", 0)))
            thal     = st.selectbox("Thalassemia", [0, 1, 2, 3],
                                    index=int(demo.get("thal", 0)),
                                    format_func=lambda x: ["Unknown","Normal","Fixed Defect","Reversible Defect"][x])

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Buttons ───────────────────────────────────────────────────────────
        b1, b2 = st.columns([1, 1], gap="small")
        with b1:
            predict_clicked = st.button("🔍 Predict Risk", type="primary", use_container_width=True)
        with b2:
            st.markdown("<div class='btn-clear'>", unsafe_allow_html=True)
            clear_clicked = st.button("🗑 Clear Form", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            if clear_clicked:
                st.session_state.demo_loaded = None
                st.session_state.last_report = None
                st.rerun()

        # ── Demo Samples ──────────────────────────────────────────────────────
        st.markdown("<p style='font-size:1.1rem;font-weight:700;color:#0f172a;margin:18px 0 8px;'>🧪 Demo Sample Cases</p>", unsafe_allow_html=True)
        st.divider()
        d1, d2, d3 = st.columns(3)
        sample_items = list(SAMPLES.items())
        with d1:
            st.markdown("<div class='btn-high'>", unsafe_allow_html=True)
            if st.button(sample_items[0][0], use_container_width=True):
                st.session_state.demo_loaded = sample_items[0][1]; st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        with d2:
            st.markdown("<div class='btn-mid'>", unsafe_allow_html=True)
            if st.button(sample_items[1][0], use_container_width=True):
                st.session_state.demo_loaded = sample_items[1][1]; st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        with d3:
            st.markdown("<div class='btn-low'>", unsafe_allow_html=True)
            if st.button(sample_items[2][0], use_container_width=True):
                st.session_state.demo_loaded = sample_items[2][1]; st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    with right:
        # ── Result Card ─────────────────────────────────────────────────────
        st.markdown("<div style='min-height:260px;'>", unsafe_allow_html=True)
        report = st.session_state.last_report
        if report:
            lvl = report["level"]
            cls = "result-high" if "High" in lvl else "result-medium" if "Medium" in lvl else "result-low"
            col = "#dc2626" if "High" in lvl else "#d97706" if "Medium" in lvl else "#16a34a"
            icon= "🔴" if "High" in lvl else "🟡" if "Medium" in lvl else "🟢"
            st.markdown(f"""
            <div class='{cls}'>
              <div class='result-label' style='color:{col};'>Prediction Result</div>
              <div class='result-pred' style='color:{col};'>{report["pred_text"]}</div>
              <div class='result-prob' style='color:{col};'>{report["prob"]*100:.1f}%</div>
              <div class='result-level' style='color:{col};'>{icon} {lvl}</div>
              <div class='result-expl'>{report["expl"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='result-none'>
              <div style='font-size:2.5rem;margin-bottom:10px;'>🫀</div>
              <div style='font-size:1rem;font-weight:600;color:#64748b;'>Awaiting Prediction</div>
              <div style='font-size:0.85rem;color:#94a3b8;margin-top:6px;'>
                Fill in patient details and click<br><b>Predict Risk</b>
              </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Recommendations ─────────────────────────────────────────────────
        if report:
            st.markdown("<div class='section-title' style='margin-top:14px;'>💊 Recommendations</div>", unsafe_allow_html=True)
            for rec in report["recs"]:
                st.markdown(f"<div class='rec-item'>• {rec}</div>", unsafe_allow_html=True)

        # ── Comparison Chart ─────────────────────────────────────────────────
        st.markdown("<div class='section-title' style='margin-top:14px;'>📊 Patient vs Normal Values</div>", unsafe_allow_html=True)
        if report:
            vals = report["values"]
            items_chart = [("BP",vals["trestbps"],120),("Chol",vals["chol"],200),
                           ("HR",vals["thalach"],170),("ST",vals["oldpeak"],1.0)]
            bars_html = "<div class='chart-wrap'><div class='bar-row'>"
            for lbl, pv, nv in items_chart:
                maxv = max(pv, nv, 0.1)
                ph = max(int((pv/maxv)*100), 6)
                nh = max(int((nv/maxv)*100), 6)
                bars_html += f"""
                <div class='bar-group'>
                  <div class='bar-pair'>
                    <div style='display:flex;flex-direction:column;align-items:center;gap:2px;'>
                      <span class='bar-val'>{pv:.0f}</span>
                      <div class='bar bar-p' style='height:{ph}px;'></div>
                    </div>
                    <div style='display:flex;flex-direction:column;align-items:center;gap:2px;'>
                      <span class='bar-val'>{nv:.0f}</span>
                      <div class='bar bar-n' style='height:{nh}px;'></div>
                    </div>
                  </div>
                  <div class='bar-lbl'>{lbl}</div>
                </div>"""
            bars_html += """</div>
            <div class='chart-legend'>
              <span><span class='leg-dot' style='background:#2563eb;'></span>Patient</span>
              <span><span class='leg-dot' style='background:#16a34a;'></span>Normal</span>
            </div></div>"""
            st.markdown(bars_html, unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align:center;color:#94a3b8;padding:20px 0;font-size:0.9rem;'>"
                        "Chart will appear after prediction</div>", unsafe_allow_html=True)

        # ── Input Guide ──────────────────────────────────────────────────────
        with st.expander("📖 Input Reference Guide"):
            guide = [
                ("Age","1 – 120 years"),("Sex","1 = Male, 0 = Female"),
                ("Chest Pain","0 Typical · 1 Atypical · 2 Non-anginal · 3 Asymptomatic"),
                ("Blood Pressure","70 – 250 mmHg  (normal < 120)"),
                ("Cholesterol","80 – 700 mg/dl  (normal < 200)"),
                ("Fasting BS","1 = > 120 mg/dl,  0 = Normal"),
                ("ECG","0 Normal · 1 ST-T abnormal · 2 LV hypertrophy"),
                ("Max Heart Rate","60 – 250 bpm"),
                ("Exercise Angina","1 = Yes,  0 = No"),
                ("ST Depression","0 – 10  (0 = normal)"),
                ("ST Slope","0 Upsloping · 1 Flat · 2 Downsloping"),
                ("Major Vessels","0 – 4 (fluoroscopy count)"),
                ("Thalassemia","0 Unknown · 1 Normal · 2 Fixed · 3 Reversible"),
            ]
            for k, v in guide:
                st.markdown(f"<div class='guide-item'><span class='guide-key'>{k}</span>"
                            f"<span class='guide-val'>{v}</span></div>", unsafe_allow_html=True)

        # ── PDF Download ──────────────────────────────────────────────────────
        if report:
            st.markdown("<div class='btn-pdf'>", unsafe_allow_html=True)
            if REPORTLAB_OK:
                pdf_bytes = make_pdf(report)
                fname = f"CardioScan_{report['name'].replace(' ','_')}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                st.download_button("📥 Download PDF Report", data=pdf_bytes,
                                   file_name=fname, mime="application/pdf", use_container_width=True)
            else:
                st.warning("Install reportlab (`pip install reportlab`) to enable PDF download.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='disclaimer'>⚕ <b>Medical Disclaimer:</b> This tool is for educational "
                    "purposes only. It is not a certified medical device. Always consult a qualified "
                    "doctor for clinical decisions.</div>", unsafe_allow_html=True)

    # ── Run Prediction ──────────────────────────────────────────────────────────
    if predict_clicked:
        errors = []
        if not p_name.strip():    errors.append("Patient name is required.")
        if not p_contact.strip(): errors.append("Contact number is required.")
        if not p_contact.replace("+","").replace("-","").replace(" ","").isdigit() and p_contact.strip():
            errors.append("Contact number should contain digits only.")
        if cp is None:    errors.append("Please select a Chest Pain Type.")
        if slope is None: errors.append("Please select an ST Slope value.")
        if errors:
            for e in errors: st.error(e)
        else:
            values = dict(age=age,sex=sex,cp=int(cp),trestbps=trestbps,chol=chol,fbs=fbs,
                          restecg=restecg,thalach=thalach,exang=exang,oldpeak=oldpeak,
                          slope=int(slope),ca=ca,thal=thal)
            pred, prob, level = predict(model, values)
            pred_text = "Heart Disease Risk Detected" if pred==1 else "No Heart Disease Risk Detected"
            alert = ("Immediate medical attention is recommended." if level=="High Risk"
                     else "Monitor health and improve lifestyle." if level=="Medium Risk"
                     else "Maintain a healthy lifestyle.")
            rpt = dict(name=p_name.strip(), contact=p_contact.strip(), values=values,
                       pred_text=pred_text, prob=prob, level=level, alert=alert,
                       recs=recommendations(values, prob), expl=explanation(values),
                       date=dt.datetime.now().strftime("%Y-%m-%d %H:%M"))
            st.session_state.last_report = rpt
            save_record(p_name.strip(), p_contact.strip(), values, pred_text, prob, level)
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "📋 Patient History":
    st.markdown("""
    <div class='hero-banner'>
      <div class='hero-icon'>📋</div>
      <div>
        <div class='hero-title'>My Prediction History</div>
        <div class='hero-sub'>Enter your name to view your own records — your data is private</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Privacy notice
    st.markdown("""
    <div style='background:#eff6ff;border:1px solid #bfdbfe;border-radius:12px;
    padding:14px 18px;margin-bottom:20px;display:flex;align-items:center;gap:12px;'>
      <span style='font-size:1.5rem;'>🔒</span>
      <div>
        <div style='font-weight:700;color:#1e40af;font-size:0.95rem;'>Your records are private</div>
        <div style='color:#3b82f6;font-size:0.84rem;margin-top:2px;'>
          Only your own records appear when you search your name. Full history is only accessible by the admin.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not os.path.exists(PATIENT_FILE):
        st.markdown("""
        <div style='text-align:center;padding:48px;background:#fff;border-radius:16px;'>
          <div style='font-size:3rem;margin-bottom:14px;'>📭</div>
          <div style='font-size:1.1rem;font-weight:600;color:#475569;'>No Records Yet</div>
          <div style='color:#94a3b8;margin-top:6px;'>Make a prediction first, then search your name here.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df_h = pd.read_csv(PATIENT_FILE)

        # Search box — user must type their name
        search = st.text_input("🔍 Enter your full name to view your records",
                               placeholder="e.g. Ahmed Raza",
                               help="Type your exact name as entered during prediction")

        if not search.strip():
            st.markdown("""
            <div style='text-align:center;padding:40px 20px;background:#fff;border-radius:16px;margin-top:10px;'>
              <div style='font-size:2.5rem;margin-bottom:12px;'>🔍</div>
              <div style='font-size:1rem;font-weight:600;color:#475569;'>Enter your name above</div>
              <div style='color:#94a3b8;font-size:0.88rem;margin-top:6px;'>
                Your prediction history will appear here once you type your name.
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            filtered = df_h[df_h["Patient Name"].astype(str).str.lower() == search.strip().lower()]
            if filtered.empty:
                st.markdown(f"""
                <div style='text-align:center;padding:40px;background:#fff;border-radius:16px;margin-top:10px;'>
                  <div style='font-size:2.5rem;margin-bottom:12px;'>🚫</div>
                  <div style='font-size:1rem;font-weight:600;color:#dc2626;'>No records found for "{search}"</div>
                  <div style='color:#94a3b8;font-size:0.88rem;margin-top:6px;'>
                    Make sure you enter your name exactly as used during prediction.
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Show only safe columns — no clinical raw numbers exposed unnecessarily
                show_cols = ["Date", "Patient Name", "Prediction", "Risk Level", "Risk Probability %"]
                show_cols = [c for c in show_cols if c in filtered.columns]
                st.markdown(f"""
                <div class='metric-row'>
                  <div class='metric-card'><div class='metric-val'>{len(filtered)}</div>
                  <div class='metric-lbl'>Your Records</div></div>
                  <div class='metric-card' style='border-top-color:#dc2626;'>
                  <div class='metric-val' style='color:#dc2626;'>
                  {int((filtered["Risk Level"]=="High Risk").sum()) if "Risk Level" in filtered.columns else 0}</div>
                  <div class='metric-lbl'>High Risk</div></div>
                  <div class='metric-card' style='border-top-color:#d97706;'>
                  <div class='metric-val' style='color:#d97706;'>
                  {int((filtered["Risk Level"]=="Medium Risk").sum()) if "Risk Level" in filtered.columns else 0}</div>
                  <div class='metric-lbl'>Medium Risk</div></div>
                  <div class='metric-card' style='border-top-color:#16a34a;'>
                  <div class='metric-val' style='color:#16a34a;'>
                  {int((filtered["Risk Level"]=="Low Risk").sum()) if "Risk Level" in filtered.columns else 0}</div>
                  <div class='metric-lbl'>Low Risk</div></div>
                </div>
                """, unsafe_allow_html=True)

                st.dataframe(
                    filtered[show_cols].iloc[::-1].reset_index(drop=True),
                    use_container_width=True, height=min(400, 80 + len(filtered)*40)
                )
                # Let user download only their own records
                st.markdown("<div class='btn-pdf'>", unsafe_allow_html=True)
                csv_bytes = filtered[show_cols].to_csv(index=False).encode()
                st.download_button("📥 Download My Records (CSV)",
                                   data=csv_bytes,
                                   file_name=f"my_records_{search.strip().replace(' ','_')}.csv",
                                   mime="text/csv", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='disclaimer' style='margin-top:20px;'>
          🔐 <b>Privacy Note:</b> Only your own records are visible here.
          Full patient data is restricted to the system administrator.
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — ADMIN
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "🔐 Admin Dashboard":
    if not st.session_state.admin_logged_in:
        st.markdown("""
        <div class='hero-banner'>
          <div class='hero-icon'>🔐</div>
          <div><div class='hero-title'>Admin Login</div>
          <div class='hero-sub'>Restricted area — authorised staff only</div></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div class='section-card' style='max-width:420px;margin:0 auto;'>", unsafe_allow_html=True)
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if u==ADMIN_USER and p==ADMIN_PASS:
                st.session_state.admin_logged_in = True; st.rerun()
            else:
                st.error("Incorrect username or password.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='hero-banner'>
          <div class='hero-icon'>📊</div>
          <div><div class='hero-title'>Admin Dashboard</div>
          <div class='hero-sub'>Dataset statistics and model performance analytics</div></div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚪 Logout"):
            st.session_state.admin_logged_in = False; st.rerun()

        # ── Model Performance ─────────────────────────────────────────────
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🎯 Model Performance (5-Fold Cross-Validation)</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='metric-row'>
          <div class='metric-card'><div class='metric-val'>{metrics['acc']}%</div><div class='metric-lbl'>Accuracy</div></div>
          <div class='metric-card'><div class='metric-val'>{metrics['prec']}%</div><div class='metric-lbl'>Precision</div></div>
          <div class='metric-card'><div class='metric-val'>{metrics['rec']}%</div><div class='metric-lbl'>Recall</div></div>
          <div class='metric-card'><div class='metric-val'>{metrics['f1']}%</div><div class='metric-lbl'>F1 Score</div></div>
          <div class='metric-card'><div class='metric-val'>{n_rows}</div><div class='metric-lbl'>Training Patients</div></div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Evaluated using 5-fold stratified cross-validation on deduplicated dataset. These are honest, non-leaked metrics.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Dataset stats
        df_raw = pd.read_csv("heart.csv")
        y_col = "target"
        hd_count = int((df_raw[y_col]<1).sum()) if y_col in df_raw.columns else 0
        nohd     = int((df_raw[y_col]>=1).sum()) if y_col in df_raw.columns else 0
        high_bp  = int((df_raw["trestbps"]>=140).sum()) if "trestbps" in df_raw else 0
        high_ch  = int((df_raw["chol"]>=240).sum()) if "chol" in df_raw else 0

        st.markdown(f"""
        <div style='display:flex;gap:14px;flex-wrap:wrap;margin-bottom:22px;'>
          <div class='admin-stat'><div class='admin-val'>{len(df_raw)}</div><div class='admin-lbl'>Total Dataset Rows</div></div>
          <div class='admin-stat'><div class='admin-val'>{hd_count}</div><div class='admin-lbl'>Heart Disease Cases</div></div>
          <div class='admin-stat'><div class='admin-val'>{nohd}</div><div class='admin-lbl'>No Heart Disease</div></div>
          <div class='admin-stat'><div class='admin-val'>{high_bp}</div><div class='admin-lbl'>High BP Patients</div></div>
          <div class='admin-stat'><div class='admin-val'>{high_ch}</div><div class='admin-lbl'>High Cholesterol</div></div>
          <div class='admin-stat'><div class='admin-val'>{round(float(df_raw["age"].mean()),1)}</div><div class='admin-lbl'>Avg Age</div></div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>📋 Model Algorithm Details</div>", unsafe_allow_html=True)
            st.markdown("""
            <div style='line-height:2;font-size:0.9rem;'>
              <div style='display:flex;justify-content:space-between;padding:8px 12px;background:#f0f4ff;border-radius:8px;margin-bottom:6px;'>
                <span style='font-weight:600;color:#0f172a;'>Algorithm</span><span style='color:#1d4ed8;font-weight:700;'>Logistic Regression</span>
              </div>
              <div style='display:flex;justify-content:space-between;padding:8px 12px;background:#f0fdf4;border-radius:8px;margin-bottom:6px;'>
                <span style='font-weight:600;color:#0f172a;'>Regularization (C)</span><span style='color:#16a34a;font-weight:700;'>0.7</span>
              </div>
              <div style='display:flex;justify-content:space-between;padding:8px 12px;background:#fff7ed;border-radius:8px;margin-bottom:6px;'>
                <span style='font-weight:600;color:#0f172a;'>Solver</span><span style='color:#d97706;font-weight:700;'>liblinear</span>
              </div>
              <div style='display:flex;justify-content:space-between;padding:8px 12px;background:#fdf4ff;border-radius:8px;margin-bottom:6px;'>
                <span style='font-weight:600;color:#0f172a;'>Class Weight</span><span style='color:#7c3aed;font-weight:700;'>Balanced</span>
              </div>
              <div style='display:flex;justify-content:space-between;padding:8px 12px;background:#f0f4ff;border-radius:8px;'>
                <span style='font-weight:600;color:#0f172a;'>Preprocessing</span><span style='color:#1d4ed8;font-weight:700;'>StandardScaler + Median Imputer</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>📈 Dataset Feature Averages</div>", unsafe_allow_html=True)
            avg_df = df_raw[["age","trestbps","chol","thalach","oldpeak"]].mean().round(1).reset_index()
            avg_df.columns = ["Feature","Average Value"]
            avg_df["Feature"] = avg_df["Feature"].map({
                "age":"Age","trestbps":"Blood Pressure","chol":"Cholesterol",
                "thalach":"Max Heart Rate","oldpeak":"ST Depression"})
            st.dataframe(avg_df, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if os.path.exists(PATIENT_FILE):
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>🏥 Prediction Records Summary</div>", unsafe_allow_html=True)
            df_p = pd.read_csv(PATIENT_FILE)
            st.dataframe(df_p.tail(50).iloc[::-1].reset_index(drop=True), use_container_width=True, height=300)
            st.download_button("📥 Download All Records", data=df_p.to_csv(index=False).encode(),
                               file_name="all_prediction_records.csv", mime="text/csv")
            st.markdown("</div>", unsafe_allow_html=True)
