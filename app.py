"""
CardioScan AI — Heart Disease Risk Prediction
Professional Streamlit Web App with:
- BMI Calculator
- Risk History Chart
- English / Urdu language toggle
- Symptom Checklist
- WhatsApp Report Sharing
- Doctor Notes in PDF
- Private patient history
- Admin dashboard
"""

import os, io, csv, datetime as dt, urllib.parse
import pandas as pd
import streamlit as st

# Google Sheets (used on Streamlit Cloud — falls back to CSV locally)
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_OK = True
except ImportError:
    GSPREAD_OK = False
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate

try:
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

st.set_page_config(page_title="CardioScan AI", page_icon="🫀", layout="wide")

FEATURES = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
DISPLAY  = {
    "age":"Age","sex":"Sex","cp":"Chest Pain Type","trestbps":"Resting BP (mmHg)",
    "chol":"Cholesterol (mg/dl)","fbs":"Fasting Blood Sugar > 120","restecg":"Resting ECG",
    "thalach":"Max Heart Rate","exang":"Exercise Angina","oldpeak":"ST Depression",
    "slope":"ST Slope","ca":"Major Vessels (0-4)","thal":"Thalassemia",
}
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

# ══════════════════════════════════════════════════════════════════════
#  TRANSLATIONS  (English / Urdu)
# ══════════════════════════════════════════════════════════════════════
T = {
    "en": {
        "nav_pred":"🏥 Prediction", "nav_hist":"📋 Patient History", "nav_admin":"🔐 Admin Dashboard",
        "hero_sub":"Heart Disease Risk Prediction System · Powered by Machine Learning",
        "fact1":"Every 33 seconds, someone in the world dies from cardiovascular disease.",
        "fact2":"80% of premature heart attacks are preventable with early screening.",
        "fact3":"High blood pressure is the #1 risk factor — affects 1 in 3 adults.",
        "fact4":"Early detection can reduce heart disease mortality by up to 50%.",
        "pat_info":"👤 Patient Information","pat_name":"Patient Name *","pat_contact":"Contact Number *",
        "pat_name_ph":"Enter full name","pat_contact_ph":"e.g. 03001234567",
        "bmi_section":"⚖️ BMI Calculator (Optional)",
        "bmi_height":"Height (cm)","bmi_weight":"Weight (kg)","bmi_result":"BMI",
        "symptom_section":"🩺 Quick Symptom Check (Optional)",
        "s1":"Do you feel chest pain or pressure?",
        "s2":"Do you get breathless climbing stairs?",
        "s3":"Do you feel dizzy or faint during activity?",
        "s4":"Do you have pain radiating to your arm or jaw?",
        "s5":"Do you have a family history of heart disease?",
        "clinical":"🩺 Clinical Data","doctor_notes":"📝 Doctor / Additional Notes (Optional)",
        "notes_ph":"Add any extra symptoms, medications, or relevant notes here…",
        "predict":"🔍 Predict Risk","clear":"🗑 Clear Form",
        "demo":"🧪 Demo Sample Cases",
        "cp_label":"Chest Pain Type *","slope_label":"ST Slope *","select":"— Select —",
        "awaiting":"Awaiting Prediction","awaiting_sub":"Fill in patient details and click Predict Risk",
        "result_label":"Prediction Result","recs":"💊 Recommendations","chart":"📊 Patient vs Normal Values",
        "chart_empty":"Chart will appear after prediction",
        "guide":"📖 Input Reference Guide","pdf":"📥 Download PDF Report",
        "whatsapp":"📲 Share via WhatsApp","disclaimer_text":"⚕ Medical Disclaimer: This tool is for educational purposes only. Not a certified medical device. Always consult a qualified doctor.",
        "hist_title":"My Prediction History","hist_sub":"Enter your name to view your own records — your data is private",
        "privacy_title":"Your records are private","privacy_sub":"Only your own records appear when you search your name.",
        "search_label":"🔍 Or search by full name","search_ph":"e.g. Ahmed Raza",
        "search_prompt":"Enter your name above","search_prompt_sub":"Your prediction history will appear once you type your name.",
        "no_records":"No records found for","no_records_sub":"Make sure you enter your name exactly as used during prediction.",
        "your_records":"Your Records","download_csv":"📥 Download My Records (CSV)",
        "risk_chart":"📈 Your Risk Over Time",
        "admin_login":"Admin Login","admin_sub":"Restricted area — authorised staff only",
        "username":"Username","password":"Password","login":"Login","logout":"🚪 Logout",
        "wrong_creds":"Incorrect username or password.",
        "no_file":"No Records Yet","no_file_sub":"Make a prediction first, then search your name here.",
        "sex_m":"Male","sex_f":"Female","yes":"Yes","no":"No",
        "cp0":"Typical Angina","cp1":"Atypical Angina","cp2":"Non-anginal Pain","cp3":"Asymptomatic",
        "ecg0":"Normal","ecg1":"ST-T Abnormality","ecg2":"LV Hypertrophy",
        "sl0":"Upsloping","sl1":"Flat","sl2":"Downsloping",
        "th0":"Unknown","th1":"Normal","th2":"Fixed Defect","th3":"Reversible Defect",
    },
    "ur": {
        "nav_pred":"🏥 پیشگوئی", "nav_hist":"📋 مریض کی تاریخ", "nav_admin":"🔐 ایڈمن",
        "hero_sub":"دل کی بیماری کے خطرے کی پیشگوئی · مشین لرننگ سے چلنے والا",
        "fact1":"دنیا میں ہر 33 سیکنڈ میں کوئی نہ کوئی دل کی بیماری سے مر جاتا ہے۔",
        "fact2":"وقت سے پہلے آنے والے 80٪ دل کے دورے قابل علاج ہیں۔",
        "fact3":"ہائی بلڈ پریشر سب سے بڑا خطرہ ہے — 3 میں سے 1 بالغ متاثر ہے۔",
        "fact4":"ابتدائی پتہ لگانے سے دل کی بیماری سے اموات 50٪ تک کم ہو سکتی ہیں۔",
        "pat_info":"👤 مریض کی معلومات","pat_name":"مریض کا نام *","pat_contact":"رابطہ نمبر *",
        "pat_name_ph":"پورا نام درج کریں","pat_contact_ph":"مثلاً 03001234567",
        "bmi_section":"⚖️ BMI کیلکولیٹر (اختیاری)",
        "bmi_height":"قد (سینٹی میٹر)","bmi_weight":"وزن (کلوگرام)","bmi_result":"BMI",
        "symptom_section":"🩺 فوری علامت کی جانچ (اختیاری)",
        "s1":"کیا آپ کو سینے میں درد یا دباؤ محسوس ہوتا ہے؟",
        "s2":"کیا سیڑھیاں چڑھتے وقت سانس پھولتا ہے؟",
        "s3":"کیا سرگرمی کے دوران چکر آتے ہیں؟",
        "s4":"کیا درد بازو یا جبڑے تک پھیلتا ہے؟",
        "s5":"کیا خاندان میں دل کی بیماری کی تاریخ ہے؟",
        "clinical":"🩺 طبی ڈیٹا","doctor_notes":"📝 ڈاکٹر / اضافی نوٹس (اختیاری)",
        "notes_ph":"یہاں اضافی علامات، ادویات یا نوٹس لکھیں…",
        "predict":"🔍 خطرے کی پیشگوئی","clear":"🗑 فارم صاف کریں",
        "demo":"🧪 ڈیمو نمونے",
        "cp_label":"سینے کے درد کی قسم *","slope_label":"ST ڈھلوان *","select":"— منتخب کریں —",
        "awaiting":"پیشگوئی کا انتظار","awaiting_sub":"مریض کی تفصیلات بھریں اور پیشگوئی کریں پر کلک کریں",
        "result_label":"پیشگوئی کا نتیجہ","recs":"💊 سفارشات","chart":"📊 مریض بمقابلہ معمول",
        "chart_empty":"پیشگوئی کے بعد چارٹ ظاہر ہوگا",
        "guide":"📖 معلوماتی رہنما","pdf":"📥 PDF رپورٹ ڈاؤن لوڈ کریں",
        "whatsapp":"📲 واٹس ایپ پر شیئر کریں","disclaimer_text":"⚕ طبی انتباہ: یہ ٹول صرف تعلیمی مقاصد کے لیے ہے۔ کسی مستند ڈاکٹر سے مشورہ کریں۔",
        "hist_title":"میری پیشگوئی کی تاریخ","hist_sub":"اپنے ریکارڈ دیکھنے کے لیے اپنا نام درج کریں",
        "privacy_title":"آپ کے ریکارڈ نجی ہیں","privacy_sub":"صرف آپ کے اپنے ریکارڈ ظاہر ہوں گے۔",
        "search_label":"🔍 یا پورا نام درج کریں","search_ph":"مثلاً احمد رضا",
        "search_prompt":"اوپر اپنا نام درج کریں","search_prompt_sub":"نام درج کرنے کے بعد آپ کی تاریخ ظاہر ہو گی۔",
        "no_records":"کوئی ریکارڈ نہیں ملا","no_records_sub":"یقینی بنائیں کہ نام بالکل وہی ہے جو پیشگوئی میں استعمال ہوا۔",
        "your_records":"آپ کے ریکارڈ","download_csv":"📥 میرے ریکارڈ ڈاؤن لوڈ کریں",
        "risk_chart":"📈 وقت کے ساتھ آپ کا خطرہ",
        "admin_login":"ایڈمن لاگ ان","admin_sub":"محدود علاقہ — صرف مجاز عملہ",
        "username":"صارف نام","password":"پاس ورڈ","login":"لاگ ان","logout":"🚪 لاگ آؤٹ",
        "wrong_creds":"غلط صارف نام یا پاس ورڈ۔",
        "no_file":"ابھی کوئی ریکارڈ نہیں","no_file_sub":"پہلے پیشگوئی کریں پھر یہاں نام تلاش کریں۔",
        "sex_m":"مرد","sex_f":"عورت","yes":"ہاں","no":"نہیں",
        "cp0":"عام انجائنا","cp1":"غیر معمولی انجائنا","cp2":"غیر انجائنا درد","cp3":"بے علامت",
        "ecg0":"معمول","ecg1":"ST-T اسامانیتا","ecg2":"LV ہائپرٹروفی",
        "sl0":"اوپر کی طرف","sl1":"چپٹا","sl2":"نیچے کی طرف",
        "th0":"نامعلوم","th1":"معمول","th2":"مستقل نقص","th3":"قابل واپسی نقص",
    },
}

# ══════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=DM+Serif+Display&display=swap');

html, body, [data-testid="stAppViewContainer"] { background:#f0f4ff !important; font-family:'Inter',sans-serif; }
[data-testid="stSidebar"] { background:linear-gradient(160deg,#0f172a 0%,#1e3a5f 100%) !important; border-right:none !important; }
[data-testid="stSidebar"] * { color:#e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { color:#cbd5e1 !important; font-size:0.95rem; }
[data-testid="stSidebar"] .stRadio [data-checked="true"] label { color:#60a5fa !important; font-weight:600; }

[data-testid="stMain"] label,[data-testid="stMain"] .stTextInput label,
[data-testid="stMain"] .stNumberInput label,[data-testid="stMain"] .stSelectbox label,
[data-testid="stMain"] p { color:#0f172a !important; font-weight:600 !important; font-size:0.88rem !important; }
[data-testid="stMain"] input { background:#ffffff !important; color:#0f172a !important; caret-color:#1d4ed8 !important; }
[data-testid="stMain"] [data-baseweb="select"] { background:#ffffff !important; border:1.5px solid #c7d2fe !important; border-radius:8px !important; }
[data-testid="stMain"] [data-baseweb="select"] * { color:#0f172a !important; background:#fff !important; }
[data-testid="stMain"] [data-baseweb="input"] { background:#ffffff !important; border:1.5px solid #c7d2fe !important; border-radius:8px !important; }

.hero-banner { background:linear-gradient(135deg,#0f172a 0%,#1e40af 60%,#0ea5e9 100%); border-radius:18px; padding:32px 44px 20px; margin-bottom:18px; box-shadow:0 8px 40px rgba(15,23,42,0.18); }
.hero-top { display:flex; align-items:center; gap:24px; margin-bottom:20px; }
.hero-icon { font-size:3.6rem; line-height:1; }
.hero-title { font-family:'DM Serif Display',serif; font-size:2.2rem; color:#fff; margin:0; letter-spacing:-0.5px; }
.hero-sub { color:#93c5fd; font-size:1rem; margin-top:5px; }
.fact-strip { display:flex; gap:12px; flex-wrap:wrap; border-top:1px solid rgba(255,255,255,0.1); padding-top:16px; }
.fact-card { flex:1; min-width:180px; background:rgba(255,255,255,0.07); border-radius:10px; padding:12px 14px; border-left:3px solid #60a5fa; }
.fact-icon { font-size:1.3rem; margin-bottom:4px; }
.fact-text { color:#e0f2fe; font-size:0.78rem; line-height:1.4; }

.metric-row { display:flex; gap:14px; margin-bottom:24px; flex-wrap:wrap; }
.metric-card { flex:1; min-width:130px; background:#fff; border-radius:14px; padding:20px 18px; box-shadow:0 2px 12px rgba(37,99,235,0.08); border-top:4px solid #2563eb; text-align:center; }
.metric-val { font-size:1.9rem; font-weight:800; color:#1e40af; }
.metric-lbl { font-size:0.78rem; color:#64748b; font-weight:600; letter-spacing:.04em; text-transform:uppercase; margin-top:2px; }

.section-title { font-size:1.1rem; font-weight:700; color:#0f172a; margin-bottom:14px; margin-top:4px; display:flex; align-items:center; gap:8px; padding-bottom:10px; border-bottom:2px solid #e8f0fe; }
.result-high   { background:linear-gradient(135deg,#fef2f2,#fee2e2); border:2px solid #fca5a5; border-radius:16px; padding:28px; text-align:center; }
.result-medium { background:linear-gradient(135deg,#fffbeb,#fef3c7); border:2px solid #fcd34d; border-radius:16px; padding:28px; text-align:center; }
.result-low    { background:linear-gradient(135deg,#f0fdf4,#dcfce7); border:2px solid #86efac; border-radius:16px; padding:28px; text-align:center; }
.result-none   { background:#f8fafc; border:2px dashed #cbd5e1; border-radius:16px; padding:28px; text-align:center; }
.result-label  { font-size:1.05rem; font-weight:600; margin-bottom:6px; }
.result-pred   { font-size:1.35rem; font-weight:800; margin-bottom:4px; }
.result-prob   { font-size:2.6rem; font-weight:900; letter-spacing:-1px; }
.result-level  { font-size:1.15rem; font-weight:700; margin-top:4px; }
.result-expl   { font-size:0.88rem; color:#475569; margin-top:12px; line-height:1.5; }

.bmi-card { background:linear-gradient(135deg,#f0f9ff,#e0f2fe); border:1px solid #bae6fd; border-radius:12px; padding:14px 18px; margin-top:10px; }
.bmi-val  { font-size:1.8rem; font-weight:800; }
.bmi-lbl  { font-size:0.82rem; font-weight:700; margin-top:2px; }

.symptom-item { background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:10px 14px; margin-bottom:6px; display:flex; align-items:center; justify-content:space-between; font-size:0.9rem; }
.symptom-warn { background:#fff7ed; border:1px solid #fed7aa; border-radius:10px; padding:10px 14px; font-size:0.84rem; color:#92400e; margin-top:8px; }

.rec-item { background:#f1f5f9; border-left:4px solid #2563eb; border-radius:0 8px 8px 0; padding:10px 14px; margin-bottom:8px; font-size:0.93rem; color:#1e293b; }
.chart-wrap { background:#f8faff; border-radius:12px; padding:18px 14px 10px; }
.bar-row { display:flex; align-items:flex-end; gap:6px; margin-bottom:16px; }
.bar-group { display:flex; flex-direction:column; align-items:center; gap:4px; flex:1; }
.bar-pair { display:flex; align-items:flex-end; gap:3px; height:110px; }
.bar { border-radius:4px 4px 0 0; min-width:22px; }
.bar-p { background:#2563eb; } .bar-n { background:#16a34a; }
.bar-lbl { font-size:0.72rem; font-weight:700; color:#475569; }
.bar-val { font-size:0.68rem; color:#64748b; }
.chart-legend { display:flex; gap:20px; justify-content:center; margin-top:8px; }
.leg-dot { width:12px; height:12px; border-radius:3px; display:inline-block; margin-right:5px; vertical-align:middle; }

.guide-item { display:flex; gap:10px; padding:7px 0; border-bottom:1px solid #f1f5f9; font-size:0.875rem; }
.guide-key { font-weight:600; color:#1e40af; min-width:130px; }
.guide-val { color:#475569; }

.stButton>button { background:linear-gradient(135deg,#1d4ed8,#2563eb) !important; color:#fff !important; border:none !important; border-radius:10px !important; font-weight:700 !important; font-size:1rem !important; padding:10px 24px !important; box-shadow:0 4px 14px rgba(37,99,235,0.3) !important; transition:all .2s !important; width:100% !important; }
.stButton>button:hover { background:linear-gradient(135deg,#1e40af,#1d4ed8) !important; transform:translateY(-1px) !important; }
.btn-clear>button { background:linear-gradient(135deg,#475569,#64748b) !important; color:#fff !important; }
/* Target clear form button by key */
button[data-testid="btn_clear_form"], [key="btn_clear_form"] button,
div:has(> button[kind="secondary"]) button {
    background:linear-gradient(135deg,#475569,#64748b) !important; color:#fff !important;
}
.btn-pdf>button   { background:linear-gradient(135deg,#059669,#10b981) !important; color:#fff !important; }
.btn-wa>button    { background:linear-gradient(135deg,#15803d,#16a34a) !important; color:#fff !important; }
.btn-high>button  { background:linear-gradient(135deg,#dc2626,#ef4444) !important; color:#fff !important; }
.btn-mid>button   { background:linear-gradient(135deg,#d97706,#f59e0b) !important; color:#fff !important; }
.btn-low>button   { background:linear-gradient(135deg,#16a34a,#22c55e) !important; color:#fff !important; }

[data-testid="stDownloadButton"]>button { background:linear-gradient(135deg,#059669,#10b981) !important; color:#fff !important; border:none !important; border-radius:10px !important; font-weight:700 !important; font-size:0.95rem !important; padding:10px 20px !important; box-shadow:0 4px 14px rgba(5,150,105,0.3) !important; width:100% !important; transition:all .2s !important; }
[data-testid="stDownloadButton"]>button:hover { background:linear-gradient(135deg,#047857,#059669) !important; color:#fff !important; }
[data-testid="stDownloadButton"]>button * { color:#fff !important; }

.admin-stat { background:#fff; border-radius:12px; padding:16px 20px; text-align:center; box-shadow:0 2px 10px rgba(0,0,0,0.06); border-top:3px solid #7c3aed; }
.admin-val  { font-size:2rem; font-weight:800; color:#7c3aed; }
.admin-lbl  { font-size:0.78rem; color:#64748b; font-weight:600; }
.disclaimer { background:#fff7ed; border:1px solid #fed7aa; border-radius:10px; padding:12px 18px; font-size:0.82rem; color:#92400e; margin-top:14px; }
.hist-chart-bar { height:18px; border-radius:4px; display:inline-block; vertical-align:middle; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Training model…")
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
    X, y = clean[FEATURES], clean["target"]
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=0.7, solver="liblinear")),
    ])
    n_splits = max(2, min(5, int(y.value_counts().min())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    sc = cross_validate(pipe, X, y, cv=cv, scoring={"acc":"accuracy","prec":"precision","rec":"recall","f1":"f1"})
    metrics = {k: round(float(sc[f"test_{k}"].mean())*100,1) for k in ["acc","prec","rec","f1"]}
    pipe.fit(X, y)
    return pipe, metrics, len(clean)

def do_predict(model, values):
    X = pd.DataFrame([values], columns=FEATURES)
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])
    level = "High Risk" if prob>=0.70 else "Medium Risk" if prob>=0.40 else "Low Risk"
    return pred, prob, level

def bmi_info(bmi):
    if bmi < 18.5: return "#3b82f6", "Underweight"
    if bmi < 25:   return "#16a34a", "Normal"
    if bmi < 30:   return "#d97706", "Overweight"
    return "#dc2626", "Obese"

def recommendations(values, prob, bmi=None, L=None):
    L = L or T["en"]
    r = []
    if prob>=0.40: r.append("Consult a cardiologist or qualified physician as soon as possible.")
    else: r.append("Maintain routine checkups and preventive care.")
    if values["trestbps"]>=140: r.append("Reduce salt intake and monitor blood pressure regularly.")
    if values["chol"]>=240:     r.append("Avoid oily/fried foods and reduce saturated fat intake.")
    if values["exang"]==1:      r.append("Avoid heavy exertion until reviewed by a doctor.")
    if bmi and bmi >= 25:       r.append(f"Work on weight management — your BMI ({bmi:.1f}) is above healthy range.")
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

# ── Google Sheets helpers ──────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_gsheet():
    """Connect to Google Sheet using Streamlit secrets. Returns sheet or None."""
    if not GSPREAD_OK: return None
    try:
        scopes = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
        creds  = Credentials.from_service_account_info(dict(st.secrets["gcp_service_account"]), scopes=scopes)
        client = gspread.authorize(creds)
        sheet  = client.open_by_key(st.secrets["sheet_id"]).sheet1
        return sheet
    except Exception:
        return None

def read_all_records():
    """Read all patient records — from Google Sheet if available, else local CSV."""
    sheet = get_gsheet()
    if sheet is not None:
        try:
            data = sheet.get_all_records()
            return pd.DataFrame(data) if data else pd.DataFrame()
        except Exception:
            pass
    # Fallback: local CSV
    if os.path.exists(PATIENT_FILE):
        try:
            return pd.read_csv(PATIENT_FILE, on_bad_lines="skip", engine="python")
        except Exception:
            pass
    return pd.DataFrame()

def generate_patient_id():
    """Generate next serial Patient ID: CARD-0001, CARD-0002 …"""
    df = read_all_records()
    if df.empty or "Patient ID" not in df.columns:
        return "CARD-0001"
    nums = []
    for pid in df["Patient ID"].dropna().astype(str):
        if pid.startswith("CARD-") and pid[5:].isdigit():
            nums.append(int(pid[5:]))
    return f"CARD-{(max(nums)+1 if nums else 1):04d}"

def save_record(name, contact, values, pred_text, prob, level, notes="", bmi=None):
    clean_notes = notes.replace("\n"," ").replace("\r"," ").strip()
    patient_id  = generate_patient_id()
    row = {"Patient ID": patient_id,
           "Date": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
           "Patient Name": name, "Contact": contact, **values,
           "BMI": round(bmi,1) if bmi else "",
           "Doctor Notes": clean_notes,
           "Prediction": pred_text,
           "Risk Probability %": round(prob*100,2),
           "Risk Level": level}
    # Try Google Sheets first
    sheet = get_gsheet()
    if sheet is not None:
        try:
            existing = sheet.get_all_records()
            if not existing:
                sheet.append_row(list(row.keys()))
            sheet.append_row([str(v) for v in row.values()])
            return patient_id
        except Exception:
            pass
    # Fallback: local CSV
    exists = os.path.exists(PATIENT_FILE)
    with open(PATIENT_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()), quoting=csv.QUOTE_ALL)
        if not exists: w.writeheader()
        w.writerow(row)
    return patient_id

def make_pdf(report):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    title_s = ParagraphStyle("T", parent=styles["Title"], textColor=rl_colors.HexColor("#dc2626"), fontSize=22, leading=28)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=rl_colors.HexColor("#1d4ed8"), spaceAfter=6)
    r = report
    story = [
        Paragraph("❤ Heart Disease Risk Prediction Report", title_s), Spacer(1,8),
        Paragraph("<b>CardioScan AI — Educational Screening System</b>", styles["Normal"]),
        Paragraph(f"<b>Report Date:</b> {r['date']}", styles["Normal"]), Spacer(1,12),
    ]
    sum_data = [["Patient ID", r.get("patient_id","—")],
                ["Patient Name", r["name"]], ["Contact Number", r["contact"]],
                ["Prediction", r["pred_text"]], ["Risk Level", r["level"]],
                ["Risk Probability", f"{r['prob']*100:.2f}%"], ["Alert", r["alert"]]]
    if r.get("bmi"): sum_data.append(["BMI", f"{r['bmi']:.1f}"])
    summary = Table(sum_data, colWidths=[160,320])
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
    story += [Paragraph("Clinical Explanation", h2), Paragraph(r["expl"], styles["Normal"]), Spacer(1,10)]
    if r.get("notes","").strip():
        story += [Paragraph("Doctor / Additional Notes", h2), Paragraph(r["notes"], styles["Normal"]), Spacer(1,10)]
    story += [Paragraph("Lifestyle & Medical Recommendations", h2)]
    for rec in r["recs"]:
        story.append(Paragraph("• " + rec, styles["Normal"]))
    story += [
        Spacer(1,20),
        Paragraph("<b>Reviewing Doctor:</b> ____________________________", styles["Normal"]),
        Spacer(1,8),
        Paragraph("<b>Date:</b> ____________________________ &nbsp; <b>Signature:</b> ____________________________", styles["Normal"]),
        Spacer(1,14),
        Paragraph("<i>Disclaimer: This is an educational ML screening report. Not a final medical diagnosis. Always consult a qualified doctor.</i>", styles["Italic"]),
    ]
    doc.build(story)
    return buf.getvalue()

# generate_patient_id() now defined above with Google Sheets support

# ══════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════
for k, v in {"last_report":None,"admin_logged_in":False,"page":"Prediction","demo_loaded":None,"lang":"en","form_key":0}.items():
    if k not in st.session_state: st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:24px 0 18px;'>
      <div style='font-size:2.6rem;'>🫀</div>
      <div style='font-family:"DM Serif Display",serif;font-size:1.5rem;color:#fff;margin-top:6px;'>CardioScan AI</div>
      <div style='color:#94a3b8;font-size:0.82rem;margin-top:4px;'>Heart Disease Risk Predictor</div>
    </div>
    <hr style='border-color:#334155;margin:0 0 18px;'>
    """, unsafe_allow_html=True)

    lang_choice = st.radio("🌐 Language / زبان", ["English", "اردو"], horizontal=True, label_visibility="collapsed")
    st.session_state.lang = "ur" if lang_choice == "اردو" else "en"
    L = T[st.session_state.lang]

    st.markdown("<hr style='border-color:#334155;margin:10px 0 14px;'>", unsafe_allow_html=True)
    page = st.radio("Navigation", [L["nav_pred"], L["nav_hist"], L["nav_admin"]], label_visibility="collapsed")
    st.session_state.page = page

    st.markdown("<hr style='border-color:#334155;margin:20px 0 14px;'>", unsafe_allow_html=True)
    model, metrics, n_rows = train_model()
    st.markdown("""
    <div style='padding:0 4px;'>

      <div style='font-size:0.78rem;font-weight:700;color:#60a5fa;letter-spacing:.06em;text-transform:uppercase;margin-bottom:10px;'>
        ❤️ Heart Health Tips
      </div>

      <div style='background:rgba(255,255,255,0.06);border-radius:10px;padding:10px 12px;margin-bottom:8px;'>
        <div style='font-size:0.78rem;color:#e2e8f0;font-weight:600;'>🩸 Know Your Numbers</div>
        <div style='font-size:0.72rem;color:#94a3b8;margin-top:4px;line-height:1.5;'>
          BP: below 120/80 mmHg<br>
          Cholesterol: below 200 mg/dl<br>
          Resting HR: 60–100 bpm
        </div>
      </div>

      <div style='background:rgba(255,255,255,0.06);border-radius:10px;padding:10px 12px;margin-bottom:8px;'>
        <div style='font-size:0.78rem;color:#e2e8f0;font-weight:600;'>⚠️ Warning Signs</div>
        <div style='font-size:0.72rem;color:#94a3b8;margin-top:4px;line-height:1.5;'>
          • Chest pain or pressure<br>
          • Shortness of breath<br>
          • Pain in arm, jaw, or back<br>
          • Dizziness or cold sweats
        </div>
      </div>

      <div style='background:rgba(255,255,255,0.06);border-radius:10px;padding:10px 12px;margin-bottom:8px;'>
        <div style='font-size:0.78rem;color:#e2e8f0;font-weight:600;'>🥗 Prevention Tips</div>
        <div style='font-size:0.72rem;color:#94a3b8;margin-top:4px;line-height:1.5;'>
          • Exercise 30 min/day<br>
          • Eat less salt & fried food<br>
          • Quit smoking<br>
          • Manage stress & sleep well
        </div>
      </div>

      <div style='background:rgba(220,38,38,0.15);border:1px solid rgba(220,38,38,0.3);border-radius:10px;padding:10px 12px;margin-bottom:8px;'>
        <div style='font-size:0.78rem;color:#fca5a5;font-weight:700;'>🚨 Emergency Pakistan</div>
        <div style='font-size:0.82rem;color:#fff;font-weight:800;margin-top:4px;'>115 — Rescue / Ambulance</div>
        <div style='font-size:0.72rem;color:#fca5a5;margin-top:2px;'>Call immediately if you have sudden chest pain or difficulty breathing.</div>
      </div>

      <div style='font-size:0.7rem;color:#475569;text-align:center;margin-top:10px;line-height:1.5;'>
        ⚕ Educational screening tool only.<br>Not a medical device.<br>Always consult a qualified doctor.
      </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  PAGE 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════
if st.session_state.page == L["nav_pred"]:

    # ── Hero banner with fact strip ────────────────────────────────────
    st.markdown(f"""
    <div class='hero-banner'>
      <div class='hero-top'>
        <div class='hero-icon'>🫀</div>
        <div>
          <div class='hero-title'>CardioScan AI</div>
          <div class='hero-sub'>{L["hero_sub"]}</div>
        </div>
      </div>
      <div class='fact-strip'>
        <div class='fact-card'><div class='fact-icon'>⏱️</div><div class='fact-text'>{L["fact1"]}</div></div>
        <div class='fact-card'><div class='fact-icon'>🛡️</div><div class='fact-text'>{L["fact2"]}</div></div>
        <div class='fact-card'><div class='fact-icon'>📊</div><div class='fact-text'>{L["fact3"]}</div></div>
        <div class='fact-card'><div class='fact-icon'>🔬</div><div class='fact-text'>{L["fact4"]}</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.55, 1], gap="large")

    with left:
        demo = st.session_state.demo_loaded or {}
        fk = st.session_state.form_key  # form key — increments on Clear to reset all widgets

        # ── Patient Information ────────────────────────────────────────
        st.markdown(f"<p class='section-title'>{L['pat_info']}</p>", unsafe_allow_html=True)
        pc1, pc2 = st.columns(2)
        p_name    = pc1.text_input(L["pat_name"],    value=demo.get("name",""),    placeholder=L["pat_name_ph"],    key=f"name_{fk}")
        p_contact = pc2.text_input(L["pat_contact"], value=demo.get("contact",""), placeholder=L["pat_contact_ph"], key=f"contact_{fk}")

        # ── BMI Calculator ─────────────────────────────────────────────
        st.markdown(f"<p class='section-title' style='margin-top:16px;'>{L['bmi_section']}</p>", unsafe_allow_html=True)
        bmi_c1, bmi_c2, bmi_c3 = st.columns(3)
        height_cm = bmi_c1.number_input(L["bmi_height"], 50, 250, 170, key=f"height_{fk}")
        weight_kg = bmi_c2.number_input(L["bmi_weight"], 10, 300, 70, key=f"weight_{fk}")
        bmi_val = weight_kg / ((height_cm/100)**2) if height_cm > 0 else None
        if bmi_val:
            bmi_color, bmi_cat = bmi_info(bmi_val)
            bmi_c3.markdown(f"""
            <div class='bmi-card'>
              <div class='bmi-val' style='color:{bmi_color};'>{bmi_val:.1f}</div>
              <div class='bmi-lbl' style='color:{bmi_color};'>{bmi_cat}</div>
            </div>""", unsafe_allow_html=True)

        # ── Symptom Checklist ──────────────────────────────────────────
        st.markdown(f"<p class='section-title' style='margin-top:16px;'>{L['symptom_section']}</p>", unsafe_allow_html=True)
        sym_keys = ["s1","s2","s3","s4","s5"]
        sym_answers = []
        s_cols = st.columns(2)
        for i, key in enumerate(sym_keys):
            with s_cols[i % 2]:
                ans = st.checkbox(L[key], key=f"sym_{i}_{fk}")
                sym_answers.append(ans)
        sym_score = sum(sym_answers)
        if sym_score >= 3:
            st.markdown(f"<div class='symptom-warn'>⚠️ You reported {sym_score}/5 warning symptoms. Please fill in clinical data carefully and consult a doctor regardless of the prediction result.</div>", unsafe_allow_html=True)
        elif sym_score >= 1:
            st.markdown(f"<div style='background:#f0fdf4;border:1px solid #86efac;border-radius:10px;padding:10px 14px;font-size:0.84rem;color:#15803d;margin-top:4px;'>ℹ️ {sym_score} symptom(s) noted. Fill in the clinical data below for a complete assessment.</div>", unsafe_allow_html=True)

        # ── Clinical Data ──────────────────────────────────────────────
        st.markdown(f"<p class='section-title' style='margin-top:16px;'>{L['clinical']}</p>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age (years)", 1, 120, int(demo.get("age",1)), key=f"age_{fk}")
            sex = st.selectbox("Sex", [1,0], index=0 if demo.get("sex",1)==1 else 1, key=f"sex_{fk}",
                               format_func=lambda x: L["sex_m"] if x else L["sex_f"])
            _cp_opts = [None,0,1,2,3]
            _cp_default = demo.get("cp",None)
            cp = st.selectbox(L["cp_label"], _cp_opts,
                              index=_cp_opts.index(_cp_default) if _cp_default in _cp_opts else 0, key=f"cp_{fk}",
                              format_func=lambda x: L["select"] if x is None else [L["cp0"],L["cp1"],L["cp2"],L["cp3"]][x])
            trestbps = st.number_input("Resting BP (mmHg)", 70, 250, int(demo.get("trestbps",70)), key=f"trestbps_{fk}")
            chol     = st.number_input("Cholesterol (mg/dl)", 80, 700, int(demo.get("chol",80)), key=f"chol_{fk}")
            fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1], index=int(demo.get("fbs",0)), key=f"fbs_{fk}",
                                    format_func=lambda x: L["yes"] if x else L["no"])
            restecg  = st.selectbox("Resting ECG", [0,1,2], index=int(demo.get("restecg",0)), key=f"restecg_{fk}",
                                    format_func=lambda x: [L["ecg0"],L["ecg1"],L["ecg2"]][x])
        with c2:
            thalach = st.number_input("Max Heart Rate (bpm)", 60, 250, int(demo.get("thalach",60)), key=f"thalach_{fk}")
            exang   = st.selectbox("Exercise-Induced Angina", [0,1], index=int(demo.get("exang",0)), key=f"exang_{fk}",
                                   format_func=lambda x: L["yes"] if x else L["no"])
            oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, float(demo.get("oldpeak",0.0)), step=0.1, key=f"oldpeak_{fk}")
            _sl_opts = [None,0,1,2]
            _sl_default = demo.get("slope",None)
            slope = st.selectbox(L["slope_label"], _sl_opts,
                                 index=_sl_opts.index(_sl_default) if _sl_default in _sl_opts else 0, key=f"slope_{fk}",
                                 format_func=lambda x: L["select"] if x is None else [L["sl0"],L["sl1"],L["sl2"]][x])
            ca   = st.selectbox("Major Vessels (0–4)", [0,1,2,3,4], index=int(demo.get("ca",0)), key=f"ca_{fk}")
            thal = st.selectbox("Thalassemia", [0,1,2,3], index=int(demo.get("thal",0)), key=f"thal_{fk}",
                                format_func=lambda x: [L["th0"],L["th1"],L["th2"],L["th3"]][x])

        # ── Doctor Notes ───────────────────────────────────────────────
        st.markdown(f"<p class='section-title' style='margin-top:16px;'>{L['doctor_notes']}</p>", unsafe_allow_html=True)
        doctor_notes = st.text_area("", placeholder=L["notes_ph"], height=90, label_visibility="collapsed", key=f"notes_{fk}")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Buttons ────────────────────────────────────────────────────
        st.markdown("""<style>
        div[data-testid="column"]:has(button[kind="secondaryFormSubmit"]),
        div[data-testid="column"]:nth-of-type(2) .stButton button:not([kind="primary"]) {
            background:linear-gradient(135deg,#475569,#64748b) !important; color:#fff !important;
        }
        </style>""", unsafe_allow_html=True)
        b1, b2 = st.columns([1,1], gap="small")
        with b1:
            predict_clicked = st.button(L["predict"], type="primary", use_container_width=True, key="btn_predict")
        with b2:
            clear_clicked = st.button(L["clear"], use_container_width=True, key="btn_clear_form")
            if clear_clicked:
                st.session_state.demo_loaded = None
                st.session_state.last_report = None
                st.session_state.form_key += 1   # forces all widgets to recreate at default values
                st.rerun()

        # ── Demo Samples ───────────────────────────────────────────────
        st.markdown(f"<p class='section-title' style='margin-top:18px;'>{L['demo']}</p>", unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        sample_items = list(SAMPLES.items())
        for col, idx, css in [(d1,0,"btn-high"),(d2,1,"btn-mid"),(d3,2,"btn-low")]:
            with col:
                st.markdown(f"<div class='{css}'>", unsafe_allow_html=True)
                if st.button(sample_items[idx][0], use_container_width=True, key=f"demo_{idx}"):
                    st.session_state.demo_loaded = sample_items[idx][1]
                    st.session_state.form_key += 1  # forces widgets to recreate with demo values
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

    # ── RIGHT COLUMN ───────────────────────────────────────────────────
    with right:
        report = st.session_state.last_report

        # Result card
        if report:
            lvl = report["level"]
            cls = "result-high" if "High" in lvl else "result-medium" if "Medium" in lvl else "result-low"
            col = "#dc2626" if "High" in lvl else "#d97706" if "Medium" in lvl else "#16a34a"
            icon= "🔴" if "High" in lvl else "🟡" if "Medium" in lvl else "🟢"
            pid = report.get("patient_id","—")
            st.markdown(f"""
            <div class='{cls}'>
              <div style='background:rgba(255,255,255,0.7);border-radius:8px;padding:5px 12px;
                   display:inline-block;margin-bottom:10px;'>
                <span style='font-size:0.72rem;font-weight:700;color:#475569;letter-spacing:.06em;text-transform:uppercase;'>Patient ID</span>
                <span style='font-size:1.1rem;font-weight:900;color:#1d4ed8;margin-left:8px;'>{pid}</span>
              </div>
              <div class='result-label' style='color:{col};'>{L["result_label"]}</div>
              <div class='result-pred' style='color:{col};'>{report["pred_text"]}</div>
              <div class='result-prob' style='color:{col};'>{report["prob"]*100:.1f}%</div>
              <div class='result-level' style='color:{col};'>{icon} {lvl}</div>
              <div class='result-expl'>{report["expl"]}</div>
              <div style='margin-top:10px;font-size:0.75rem;color:#475569;background:rgba(255,255,255,0.6);
                   border-radius:6px;padding:6px 10px;'>
                📋 Save your Patient ID <b>{pid}</b> to view your history anytime.
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-none'>
              <div style='font-size:2.5rem;margin-bottom:10px;'>🫀</div>
              <div style='font-size:1rem;font-weight:600;color:#64748b;'>{L["awaiting"]}</div>
              <div style='font-size:0.85rem;color:#94a3b8;margin-top:6px;'>{L["awaiting_sub"]}</div>
            </div>""", unsafe_allow_html=True)

        # Recommendations
        if report:
            st.markdown(f"<p class='section-title' style='margin-top:14px;'>{L['recs']}</p>", unsafe_allow_html=True)
            for rec in report["recs"]:
                st.markdown(f"<div class='rec-item'>• {rec}</div>", unsafe_allow_html=True)

        # Comparison chart
        st.markdown(f"<p class='section-title' style='margin-top:14px;'>{L['chart']}</p>", unsafe_allow_html=True)
        if report:
            vals = report["values"]
            items_chart = [("BP",vals["trestbps"],120),("Chol",vals["chol"],200),
                           ("HR",vals["thalach"],170),("ST dep",vals["oldpeak"],1.0)]
            bars_html = "<div class='chart-wrap'><div class='bar-row'>"
            for lbl, pv, nv in items_chart:
                maxv = max(pv,nv,0.1)
                ph = max(int((pv/maxv)*100),6)
                nh = max(int((nv/maxv)*100),6)
                bars_html += f"""<div class='bar-group'>
                  <div class='bar-pair'>
                    <div style='display:flex;flex-direction:column;align-items:center;gap:2px;'>
                      <span class='bar-val'>{pv:.0f}</span><div class='bar bar-p' style='height:{ph}px;'></div>
                    </div>
                    <div style='display:flex;flex-direction:column;align-items:center;gap:2px;'>
                      <span class='bar-val'>{nv:.0f}</span><div class='bar bar-n' style='height:{nh}px;'></div>
                    </div>
                  </div><div class='bar-lbl'>{lbl}</div></div>"""
            bars_html += """</div><div class='chart-legend'>
              <span><span class='leg-dot' style='background:#2563eb;'></span>Patient</span>
              <span><span class='leg-dot' style='background:#16a34a;'></span>Normal</span>
            </div></div>"""
            st.markdown(bars_html, unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align:center;color:#94a3b8;padding:20px 0;font-size:0.9rem;'>{L['chart_empty']}</div>", unsafe_allow_html=True)

        # Input Guide
        with st.expander(L["guide"]):
            guide = [
                ("Age","1–120 years"),("Sex","Male=1, Female=0"),
                ("Chest Pain","0 Typical · 1 Atypical · 2 Non-anginal · 3 Asymptomatic"),
                ("Blood Pressure","70–250 mmHg (normal <120)"),("Cholesterol","80–700 mg/dl (normal <200)"),
                ("Fasting BS","Yes = >120 mg/dl"),("ECG","0 Normal · 1 ST-T · 2 LV Hypertrophy"),
                ("Max Heart Rate","60–250 bpm"),("Exercise Angina","Yes/No"),
                ("ST Depression","0–10 (0=normal)"),("ST Slope","0 Up · 1 Flat · 2 Down"),
                ("Major Vessels","0–4 (fluoroscopy)"),("Thalassemia","1 Normal · 2 Fixed · 3 Reversible"),
            ]
            for k,v in guide:
                st.markdown(f"<div class='guide-item'><span class='guide-key'>{k}</span><span class='guide-val'>{v}</span></div>", unsafe_allow_html=True)

        # PDF + WhatsApp
        if report:
            st.markdown("<div class='btn-pdf'>", unsafe_allow_html=True)
            try:
                pdf_bytes = make_pdf(report)
                fname = f"CardioScan_{report['name'].replace(' ','_')}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                st.download_button(L["pdf"], data=pdf_bytes, file_name=fname, mime="application/pdf", use_container_width=True)
            except Exception as pdf_err:
                st.warning(f"PDF unavailable: {pdf_err}")
            st.markdown("</div>", unsafe_allow_html=True)

            # WhatsApp share button
            wa_text = (f"*CardioScan AI — Heart Disease Risk Report*\n\n"
                       f"Patient: {report['name']}\n"
                       f"Result: {report['pred_text']}\n"
                       f"Risk Level: {report['level']}\n"
                       f"Probability: {report['prob']*100:.1f}%\n\n"
                       f"⚕ This is a screening result only. Please consult a qualified doctor.\n"
                       f"Generated by CardioScan AI")
            wa_url = "https://wa.me/?text=" + urllib.parse.quote(wa_text)
            st.markdown(f"<div class='btn-wa'><a href='{wa_url}' target='_blank' style='display:block;'>"
                        f"<button style='width:100%;background:linear-gradient(135deg,#15803d,#16a34a);color:#fff;border:none;border-radius:10px;font-weight:700;font-size:1rem;padding:10px;cursor:pointer;'>{L['whatsapp']}</button>"
                        f"</a></div>", unsafe_allow_html=True)

        st.markdown(f"<div class='disclaimer'>{L['disclaimer_text']}</div>", unsafe_allow_html=True)

    # ── Run Prediction ─────────────────────────────────────────────────
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
            pred, prob, level = do_predict(model, values)
            pred_text = "Heart Disease Risk Detected" if pred==1 else "No Heart Disease Risk Detected"
            alert = ("Immediate medical attention recommended." if level=="High Risk"
                     else "Monitor health and improve lifestyle." if level=="Medium Risk"
                     else "Maintain a healthy lifestyle.")
            rpt = dict(name=p_name.strip(), contact=p_contact.strip(), values=values,
                       pred_text=pred_text, prob=prob, level=level, alert=alert,
                       recs=recommendations(values, prob, bmi_val),
                       expl=explanation(values),
                       notes=doctor_notes.strip(),
                       bmi=bmi_val,
                       date=dt.datetime.now().strftime("%Y-%m-%d %H:%M"))
            patient_id = save_record(p_name.strip(), p_contact.strip(), values, pred_text, prob, level, doctor_notes.strip(), bmi_val)
            rpt["patient_id"] = patient_id
            st.session_state.last_report = rpt
            st.rerun()

# ══════════════════════════════════════════════════════════════════════
#  PAGE 2 — PATIENT HISTORY
# ══════════════════════════════════════════════════════════════════════
elif st.session_state.page == L["nav_hist"]:
    st.markdown(f"""
    <div class='hero-banner'>
      <div class='hero-top'>
        <div class='hero-icon'>📋</div>
        <div>
          <div class='hero-title'>{L["hist_title"]}</div>
          <div class='hero-sub'>{L["hist_sub"]}</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style='background:#eff6ff;border:1px solid #bfdbfe;border-radius:12px;padding:14px 18px;margin-bottom:20px;display:flex;align-items:center;gap:12px;'>
      <span style='font-size:1.5rem;'>🔒</span>
      <div>
        <div style='font-weight:700;color:#1e40af;font-size:0.95rem;'>{L["privacy_title"]}</div>
        <div style='color:#3b82f6;font-size:0.84rem;margin-top:2px;'>{L["privacy_sub"]}</div>
      </div>
    </div>""", unsafe_allow_html=True)

    df_h = read_all_records()
    if df_h.empty:
        st.markdown(f"""
        <div style='text-align:center;padding:48px;background:#fff;border-radius:16px;'>
          <div style='font-size:3rem;margin-bottom:14px;'>📭</div>
          <div style='font-size:1.1rem;font-weight:600;color:#475569;'>{L["no_file"]}</div>
          <div style='color:#94a3b8;margin-top:6px;'>{L["no_file_sub"]}</div>
        </div>""", unsafe_allow_html=True)
    else:
        # Search by Patient ID (primary) or Name (secondary)
        col_id, col_name, col_btn = st.columns([1,1,0.4], gap="small")
        search_id   = col_id.text_input("🪪 Search by Patient ID", placeholder="e.g. CARD-0001").strip()
        search_name = col_name.text_input(L["search_label"], placeholder=L["search_ph"]).strip()
        col_btn.markdown("<br>", unsafe_allow_html=True)
        search_clicked = col_btn.button("🔍 Search", use_container_width=True, type="primary")

        # Search fires on button click OR on Enter (when either field has value)
        do_search = search_clicked or bool(search_id or search_name)

        if not do_search or (not search_id and not search_name):
            st.markdown(f"""
            <div style='background:#eff6ff;border:1px solid #bfdbfe;border-radius:12px;
                 padding:16px 20px;margin:12px 0;display:flex;gap:14px;align-items:flex-start;'>
              <span style='font-size:1.6rem;'>🪪</span>
              <div>
                <div style='font-weight:700;color:#1e40af;font-size:0.95rem;'>How to find your records</div>
                <div style='color:#3b82f6;font-size:0.84rem;margin-top:4px;line-height:1.6;'>
                  • Enter your <b>Patient ID</b> (e.g. CARD-0001) shown on your result screen or PDF report<br>
                  • OR enter your <b>full name</b> as typed during prediction<br>
                  • If multiple people share the same name, use the Patient ID for exact results
                </div>
              </div>
            </div>
            <div style='text-align:center;padding:32px;background:#fff;border-radius:16px;margin-top:10px;'>
              <div style='font-size:2.5rem;margin-bottom:12px;'>🔍</div>
              <div style='font-size:1rem;font-weight:600;color:#475569;'>{L["search_prompt"]}</div>
              <div style='color:#94a3b8;font-size:0.88rem;margin-top:6px;'>{L["search_prompt_sub"]}</div>
            </div>""", unsafe_allow_html=True)
        else:
            # ID search is exact match; name search is case-insensitive exact match
            if search_id and "Patient ID" in df_h.columns:
                filtered = df_h[df_h["Patient ID"].astype(str).str.upper() == search_id.upper()]
            elif search_name:
                filtered = df_h[df_h["Patient Name"].astype(str).str.lower() == search_name.lower()]
            else:
                filtered = pd.DataFrame()
            if filtered.empty:
                st.markdown(f"""
                <div style='text-align:center;padding:40px;background:#fff;border-radius:16px;margin-top:10px;'>
                  <div style='font-size:2.5rem;margin-bottom:12px;'>🚫</div>
                  <div style='font-size:1rem;font-weight:600;color:#dc2626;'>{L["no_records"]} "{search}"</div>
                  <div style='color:#94a3b8;font-size:0.88rem;margin-top:6px;'>{L["no_records_sub"]}</div>
                </div>""", unsafe_allow_html=True)
            else:
                high = int((filtered["Risk Level"]=="High Risk").sum()) if "Risk Level" in filtered.columns else 0
                med  = int((filtered["Risk Level"]=="Medium Risk").sum()) if "Risk Level" in filtered.columns else 0
                low  = int((filtered["Risk Level"]=="Low Risk").sum()) if "Risk Level" in filtered.columns else 0
                st.markdown(f"""
                <div class='metric-row'>
                  <div class='metric-card'><div class='metric-val'>{len(filtered)}</div><div class='metric-lbl'>{L["your_records"]}</div></div>
                  <div class='metric-card' style='border-top-color:#dc2626;'><div class='metric-val' style='color:#dc2626;'>{high}</div><div class='metric-lbl'>High Risk</div></div>
                  <div class='metric-card' style='border-top-color:#d97706;'><div class='metric-val' style='color:#d97706;'>{med}</div><div class='metric-lbl'>Medium Risk</div></div>
                  <div class='metric-card' style='border-top-color:#16a34a;'><div class='metric-val' style='color:#16a34a;'>{low}</div><div class='metric-lbl'>Low Risk</div></div>
                </div>""", unsafe_allow_html=True)

                show_cols = [c for c in ["Patient ID","Date","Patient Name","Contact","Prediction","Risk Level","Risk Probability %","BMI","Doctor Notes"] if c in filtered.columns]
                st.dataframe(filtered[show_cols].iloc[::-1].reset_index(drop=True),
                             use_container_width=True, height=min(400,80+len(filtered)*40))

                # Risk over time chart
                if "Risk Probability %" in filtered.columns and len(filtered) > 1:
                    st.markdown(f"<p class='section-title' style='margin-top:16px;'>{L['risk_chart']}</p>", unsafe_allow_html=True)
                    chart_df = filtered[["Date","Risk Probability %"]].copy().sort_values("Date").reset_index(drop=True)
                    chart_html = "<div style='background:#f8faff;border-radius:12px;padding:16px;'>"
                    max_prob = float(chart_df["Risk Probability %"].max())
                    for _, row in chart_df.iterrows():
                        prob_val = float(row["Risk Probability %"])
                        bar_w = int((prob_val / max(max_prob,1)) * 260)
                        bar_col = "#dc2626" if prob_val>=70 else "#d97706" if prob_val>=40 else "#16a34a"
                        chart_html += f"""
                        <div style='display:flex;align-items:center;gap:10px;margin-bottom:8px;font-size:0.82rem;'>
                          <span style='color:#64748b;min-width:120px;'>{str(row["Date"])[:16]}</span>
                          <div class='hist-chart-bar' style='width:{bar_w}px;background:{bar_col};'></div>
                          <span style='font-weight:700;color:{bar_col};'>{prob_val:.1f}%</span>
                        </div>"""
                    chart_html += "</div>"
                    st.markdown(chart_html, unsafe_allow_html=True)

                st.markdown("<div class='btn-pdf'>", unsafe_allow_html=True)
                st.download_button(L["download_csv"],
                                   data=filtered[show_cols].to_csv(index=False).encode(),
                                   file_name=f"my_records_{(search_id or search_name).strip().replace(' ','_')}.csv",
                                   mime="text/csv", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""<div class='disclaimer' style='margin-top:20px;'>
          🔐 <b>Privacy Note:</b> Only your own records are visible here. Full data is restricted to the administrator.
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  PAGE 3 — ADMIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════
elif st.session_state.page == L["nav_admin"]:
    if not st.session_state.admin_logged_in:
        st.markdown(f"""
        <div class='hero-banner'>
          <div class='hero-top'><div class='hero-icon'>🔐</div>
          <div><div class='hero-title'>{L["admin_login"]}</div>
          <div class='hero-sub'>{L["admin_sub"]}</div></div></div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<div class='section-card' style='max-width:420px;margin:0 auto;'>", unsafe_allow_html=True)
        u = st.text_input(L["username"])
        p = st.text_input(L["password"], type="password")
        if st.button(L["login"]):
            if u==ADMIN_USER and p==ADMIN_PASS:
                st.session_state.admin_logged_in=True; st.rerun()
            else: st.error(L["wrong_creds"])
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='hero-banner'>
          <div class='hero-top'><div class='hero-icon'>📊</div>
          <div><div class='hero-title'>Admin Dashboard</div>
          <div class='hero-sub'>Dataset statistics and model performance analytics</div></div></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div class='btn-clear'>", unsafe_allow_html=True)
        if st.button(L["logout"]): st.session_state.admin_logged_in=False; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🎯 Model Performance (5-Fold Cross-Validation)</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='metric-row'>
          <div class='metric-card'><div class='metric-val'>{metrics['acc']}%</div><div class='metric-lbl'>Accuracy</div></div>
          <div class='metric-card'><div class='metric-val'>{metrics['prec']}%</div><div class='metric-lbl'>Precision</div></div>
          <div class='metric-card'><div class='metric-val'>{metrics['rec']}%</div><div class='metric-lbl'>Recall</div></div>
          <div class='metric-card'><div class='metric-val'>{metrics['f1']}%</div><div class='metric-lbl'>F1 Score</div></div>
          <div class='metric-card'><div class='metric-val'>{n_rows}</div><div class='metric-lbl'>Training Patients</div></div>
        </div>""", unsafe_allow_html=True)
        st.caption("5-fold stratified cross-validation on 1,228 deduplicated patients from 5 sources.")
        st.markdown("</div>", unsafe_allow_html=True)

        df_raw = pd.read_csv("heart.csv")
        hd_count = int((df_raw["target"]>=1).sum())
        nohd     = int((df_raw["target"]<1).sum())
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
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>📋 Model Algorithm Details</div>", unsafe_allow_html=True)
            st.markdown("""<div style='line-height:2;font-size:0.9rem;'>
              <div style='display:flex;justify-content:space-between;padding:8px 12px;background:#f0f4ff;border-radius:8px;margin-bottom:6px;'><span style='font-weight:600;color:#0f172a;'>Algorithm</span><span style='color:#1d4ed8;font-weight:700;'>Logistic Regression</span></div>
              <div style='display:flex;justify-content:space-between;padding:8px 12px;background:#f0fdf4;border-radius:8px;margin-bottom:6px;'><span style='font-weight:600;color:#0f172a;'>Regularization (C)</span><span style='color:#16a34a;font-weight:700;'>0.7</span></div>
              <div style='display:flex;justify-content:space-between;padding:8px 12px;background:#fff7ed;border-radius:8px;margin-bottom:6px;'><span style='font-weight:600;color:#0f172a;'>Solver</span><span style='color:#d97706;font-weight:700;'>liblinear</span></div>
              <div style='display:flex;justify-content:space-between;padding:8px 12px;background:#fdf4ff;border-radius:8px;margin-bottom:6px;'><span style='font-weight:600;color:#0f172a;'>Class Weight</span><span style='color:#7c3aed;font-weight:700;'>Balanced</span></div>
              <div style='display:flex;justify-content:space-between;padding:8px 12px;background:#f0f4ff;border-radius:8px;'><span style='font-weight:600;color:#0f172a;'>Preprocessing</span><span style='color:#1d4ed8;font-weight:700;'>StandardScaler + Median Imputer</span></div>
            </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>📈 Dataset Feature Averages</div>", unsafe_allow_html=True)
            avg_df = df_raw[["age","trestbps","chol","thalach","oldpeak"]].mean().round(1).reset_index()
            avg_df.columns = ["Feature","Average Value"]
            avg_df["Feature"] = avg_df["Feature"].map({"age":"Age","trestbps":"Blood Pressure","chol":"Cholesterol","thalach":"Max Heart Rate","oldpeak":"ST Depression"})
            st.dataframe(avg_df, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

        df_p = read_all_records()
        if not df_p.empty:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>🏥 All Prediction Records</div>", unsafe_allow_html=True)
            st.dataframe(df_p.tail(100).iloc[::-1].reset_index(drop=True), use_container_width=True, height=320)
            st.download_button("📥 Download All Records", data=df_p.to_csv(index=False).encode(),
                               file_name="all_prediction_records.csv", mime="text/csv")
            st.markdown("</div>", unsafe_allow_html=True)
