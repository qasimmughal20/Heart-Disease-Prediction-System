"""
Premium Heart Disease Risk Prediction System

How to use:
1. Put this file in the same folder as heart.csv.
2. Run: python premium_heart_disease_project.py
3. Admin login: username = admin, password = admin123

Features added for presentation:
- Patient/Admin home page
- Secure admin login
- Modern colorful GUI
- Predict, save, clear, demo sample and PDF buttons
- Risk probability percentage
- Low/Medium/High color result card
- Smart popup alerts
- Patient vs normal indicator chart
- Professional PDF report
- Patient history CSV and history viewer
- Admin dashboard with statistics and visual analytics
- Dynamic lifestyle recommendations
- Simple AI-style explanation
- Optional voice output if pyttsx3 is installed
- Medical-safe model evaluation: duplicate removal, stratified cross-validation, no training/test leakage
- Patient predictions are saved in history, not appended back into the training dataset
"""

import os
import csv
import pickle
import datetime as dt
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

try:
    import pyttsx3
    VOICE_AVAILABLE = True
except Exception:
    VOICE_AVAILABLE = False

APP_TITLE = "Heart Disease Risk Prediction System"
DATASET_CANDIDATES = ["heart.csv", "Heart.csv", "heart_disease.csv", "Heart_Disease.csv"]
MODEL_FILE = "heart_disease_model.pkl"
PATIENT_RECORD_FILE = "heart_patient_prediction_records.csv"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
ADMIN_USER = "admin"
ADMIN_PASSWORD = "admin123"

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

DISPLAY_NAMES = {
    "age": "Age",
    "sex": "Sex (1=Male, 0=Female)",
    "cp": "Chest Pain Type (0-3)",
    "trestbps": "Resting Blood Pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting Blood Sugar >120 (1/0)",
    "restecg": "Resting ECG (0-2)",
    "thalach": "Max Heart Rate",
    "exang": "Exercise Angina (1/0)",
    "oldpeak": "ST Depression",
    "slope": "ST Slope (0-2)",
    "ca": "Major Vessels (0-4)",
    "thal": "Thalassemia (0-3)",
}

RANGES = {
    "age": (1, 120),
    "sex": (0, 1),
    "cp": (0, 3),
    "trestbps": (70, 250),
    "chol": (80, 700),
    "fbs": (0, 1),
    "restecg": (0, 2),
    "thalach": (60, 250),
    "exang": (0, 1),
    "oldpeak": (0, 10),
    "slope": (0, 2),
    "ca": (0, 4),
    "thal": (0, 3),
}

ALIASES = {
    "target": ["target", "HeartDisease", "heartdisease", "output", "num", "condition"],
    "trestbps": ["trestbps", "restingbp", "resting_blood_pressure", "RestingBP"],
    "chol": ["chol", "cholesterol", "serum_cholesterol", "Cholesterol"],
    "thalach": ["thalach", "maxhr", "max_heart_rate", "MaxHR"],
    "oldpeak": ["oldpeak", "st_depression", "Oldpeak"],
}

COLORS = {
    "navy": "#0f172a", "blue": "#2563eb", "sky": "#0ea5e9", "green": "#16a34a",
    "red": "#dc2626", "orange": "#f97316", "purple": "#7c3aed", "gold": "#ca8a04",
    "pink": "#db2777", "cyan": "#06b6d4", "bg": "#eef6ff", "card": "#ffffff", "muted": "#64748b",
    "yellow": "#eab308", "light_green": "#dcfce7", "light_yellow": "#fef9c3", "light_red": "#fee2e2"
}

NORMAL_REFERENCE = {
    "trestbps": 120,
    "chol": 200,
    "thalach": 170,
    "oldpeak": 1,
}

SAMPLES = {
    # These are medically realistic examples. The GUI also re-checks them with the trained model
    # and assigns the High/Medium/Low demo buttons according to actual predicted probability.
    "High Risk Demo": {
        "Patient Name": "Ali Khan", "Contact Number": "03001234567", "age": 63, "sex": 1, "cp": 0,
        "trestbps": 180, "chol": 320, "fbs": 1, "restecg": 2, "thalach": 95,
        "exang": 1, "oldpeak": 6.0, "slope": 2, "ca": 4, "thal": 3,
    },
    "Medium Risk Demo": {
        "Patient Name": "Sara Ahmed", "Contact Number": "03111234567", "age": 48, "sex": 0, "cp": 1,
        "trestbps": 140, "chol": 220, "fbs": 0, "restecg": 1, "thalach": 150,
        "exang": 0, "oldpeak": 1.8, "slope": 1, "ca": 1, "thal": 2,
    },
    "Low Risk Demo": {
        "Patient Name": "Ahmed Raza", "Contact Number": "03221234567", "age": 29, "sex": 1, "cp": 3,
        "trestbps": 110, "chol": 160, "fbs": 0, "restecg": 0, "thalach": 190,
        "exang": 0, "oldpeak": 0.0, "slope": 0, "ca": 0, "thal": 1,
    },
}


def find_dataset():
    for name in DATASET_CANDIDATES:
        if os.path.exists(name):
            return name
    for name in os.listdir("."):
        if name.lower().endswith(".csv") and "heart" in name.lower():
            return name
    for name in os.listdir("."):
        if name.lower().endswith(".csv"):
            return name
    return None


def normalize_columns(df):
    rename = {}
    lower_map = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for feature in FEATURES:
        key = feature.lower().replace("_", "")
        if key in lower_map:
            rename[lower_map[key]] = feature
    for target_name in ALIASES["target"]:
        key = target_name.lower().replace(" ", "").replace("_", "")
        if key in lower_map:
            rename[lower_map[key]] = "target"
    for canonical, aliases in ALIASES.items():
        if canonical == "target":
            continue
        for alias in aliases:
            key = alias.lower().replace(" ", "").replace("_", "")
            if key in lower_map:
                rename[lower_map[key]] = canonical
    return df.rename(columns=rename)


def target_column(df):
    for col in ["target", "HeartDisease", "output", "condition", "num"]:
        if col in df.columns:
            return col
    return None


class ModelManager:
    def __init__(self):
        self.dataset_path = find_dataset()
        self.df = None
        self.model = None
        self.metrics = {"Accuracy": 0, "Precision": 0, "Recall": 0, "F1 Score": 0}
        self.evaluation_note = "Not evaluated"
        self.load_or_train()

    def load_or_train(self):
        if not self.dataset_path:
            raise FileNotFoundError("No heart CSV dataset found. Put heart.csv in this folder.")
        self.df = normalize_columns(pd.read_csv(self.dataset_path))
        missing = [c for c in FEATURES if c not in self.df.columns]
        if missing:
            raise ValueError("Dataset is missing required heart-disease columns: " + ", ".join(missing))
        y_col = target_column(self.df)
        if not y_col:
            raise ValueError("Dataset must contain a target/output/HeartDisease column.")
        # Build a clean training table and remove exact duplicate rows before evaluation.
        # This avoids train/test leakage, which can otherwise create fake 100% medical metrics.
        clean = self.df[FEATURES + [y_col]].copy()
        for col in FEATURES + [y_col]:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")
        clean = clean.dropna(subset=[y_col])
        clean[y_col] = (clean[y_col].astype(float) > 0).astype(int)
        clean = clean.drop_duplicates(subset=FEATURES + [y_col]).reset_index(drop=True)

        X = clean[FEATURES]
        y = clean[y_col]

        # Logistic Regression is intentionally used here instead of an unrestricted tree model.
        # It is less likely to memorize the dataset and gives more realistic medical screening metrics.
        self.model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=0.7, solver="liblinear")),
        ])

        if len(clean) > 30 and y.nunique() > 1:
            min_class_count = int(y.value_counts().min())
            n_splits = max(2, min(5, min_class_count))
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            scores = cross_validate(
                self.model, X, y, cv=cv,
                scoring={"acc": "accuracy", "prec": "precision", "rec": "recall", "f1": "f1"},
                error_score="raise"
            )
            self.metrics = {
                "Accuracy": round(float(scores["test_acc"].mean()) * 100, 2),
                "Precision": round(float(scores["test_prec"].mean()) * 100, 2),
                "Recall": round(float(scores["test_rec"].mean()) * 100, 2),
                "F1 Score": round(float(scores["test_f1"].mean()) * 100, 2),
            }
            self.evaluation_note = f"{n_splits}-Fold Stratified CV on deduplicated data"
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y if y.nunique() > 1 else None)
            self.model.fit(X_train, y_train)
            pred = self.model.predict(X_test)
            self.metrics = {
                "Accuracy": round(accuracy_score(y_test, pred) * 100, 2),
                "Precision": round(precision_score(y_test, pred, zero_division=0) * 100, 2),
                "Recall": round(recall_score(y_test, pred, zero_division=0) * 100, 2),
                "F1 Score": round(f1_score(y_test, pred, zero_division=0) * 100, 2),
            }
            self.evaluation_note = "Holdout test split"

        # Fit final deployable model only after evaluation has been calculated from unseen folds/split.
        self.model.fit(X, y)
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(self.model, f)

    def predict(self, values):
        X = pd.DataFrame([values], columns=FEATURES)
        prediction = int(self.model.predict(X)[0])
        probability = float(self.model.predict_proba(X)[0][1]) if hasattr(self.model[-1], "predict_proba") else (0.75 if prediction else 0.25)
        if probability >= 0.70:
            level = "High Risk"
        elif probability >= 0.40:
            level = "Medium Risk"
        else:
            level = "Low Risk"
        return prediction, probability, level

    def append_dataset(self, values, prediction):
        # IMPORTANT: do not append model-predicted labels back into heart.csv.
        # Adding predictions as if they were real medical labels can cause data leakage
        # and unsafe inflated accuracy in future runs. Patient predictions are stored
        # separately in PATIENT_RECORD_FILE inside predict_patient().
        return

    def stats(self):
        df = self.df
        y_col = target_column(df)
        disease = int(((df[y_col].astype(float) > 0)).sum()) if y_col else 0
        no_disease = int(len(df) - disease)
        high_bp = int((df["trestbps"] >= 140).sum()) if "trestbps" in df else 0
        high_chol = int((df["chol"] >= 240).sum()) if "chol" in df else 0
        return {
            "Total Patients": len(df), "Heart Disease": disease, "No Heart Disease": no_disease,
            "High BP Count": high_bp, "High Cholesterol": high_chol,
            "Average Age": round(float(df["age"].mean()), 1), "Average Chol": round(float(df["chol"].mean()), 1),
            **self.metrics,
        }


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1380x850")
        self.minsize(1120, 720)
        self.configure(bg=COLORS["bg"])
        self.resizable(True, True)
        self.manager = ModelManager()
        self.patient_vars = {}
        self.last_report = None
        self.result_card = None
        self.chart_canvas = None
        self.show_home()

    def clear(self):
        for w in self.winfo_children():
            w.destroy()

    def header(self, title, subtitle, show_logout=False):
        frame = tk.Frame(self, bg=COLORS["navy"], height=128)
        frame.pack(fill="x")
        frame.pack_propagate(False)
        tk.Label(frame, text="❤  " + title, bg=COLORS["navy"], fg="white", font=("Segoe UI", 30, "bold")).pack(pady=(18, 4))
        tk.Label(frame, text=subtitle, bg=COLORS["navy"], fg="#bfdbfe", font=("Segoe UI", 14)).pack()
        if show_logout:
            tk.Button(frame, text="⎋  Logout", command=self.show_home, bg=COLORS["red"], fg="white", relief="flat", font=("Segoe UI", 12, "bold"), padx=24, pady=11, cursor="hand2").place(relx=.885, rely=.30)

    def make_button(self, parent, text, color, command, size=13):
        btn = tk.Button(parent, text=text, command=command, bg=color, fg="white", relief="flat", cursor="hand2", activebackground=color, activeforeground="white", font=("Segoe UI", size, "bold"), padx=18, pady=12)
        btn.bind("<Enter>", lambda e: btn.configure(relief="raised"))
        btn.bind("<Leave>", lambda e: btn.configure(relief="flat"))
        return btn

    def make_scrollable_area(self, parent, bg=None):
        """Create a reusable scrollable frame with mouse-wheel support."""
        bg = bg or COLORS["bg"]
        canvas = tk.Canvas(parent, bg=bg, highlightthickness=0)
        y_scroll = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg=bg)
        window_id = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

        def _configure_frame(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _configure_canvas(event):
            canvas.itemconfigure(window_id, width=event.width)

        def _on_mousewheel(event):
            if event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")
            else:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        scroll_frame.bind("<Configure>", _configure_frame)
        canvas.bind("<Configure>", _configure_canvas)
        for widget in (canvas, scroll_frame):
            widget.bind("<MouseWheel>", _on_mousewheel)
            widget.bind("<Button-4>", _on_mousewheel)
            widget.bind("<Button-5>", _on_mousewheel)

        canvas.configure(yscrollcommand=y_scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        y_scroll.pack(side="right", fill="y")
        return canvas, scroll_frame

    def bind_mousewheel_to_canvas(self, canvas, *widgets):
        """Allow scrolling when the pointer is over entries/labels inside a canvas."""
        def _on_mousewheel(event):
            if event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")
            else:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        for widget in widgets:
            try:
                widget.bind("<MouseWheel>", _on_mousewheel)
                widget.bind("<Button-4>", _on_mousewheel)
                widget.bind("<Button-5>", _on_mousewheel)
            except Exception:
                pass

    def show_home(self):
        self.clear()
        self.header("Heart Disease Risk Prediction System", "Premium medical prediction, reporting, analytics, and administration portal")
        body = tk.Frame(self, bg=COLORS["bg"])
        body.pack(fill="both", expand=True, padx=75, pady=55)
        card = tk.Frame(body, bg=COLORS["card"], highlightbackground="#bfdbfe", highlightthickness=2)
        card.pack(fill="both", expand=True)
        tk.Label(card, text="Choose Your Portal", bg="white", fg=COLORS["navy"], font=("Segoe UI", 32, "bold")).pack(pady=(55, 15))
        tk.Label(card, text="Patient view predicts heart risk and creates a PDF report. Admin view shows dataset and model analytics.", bg="white", fg=COLORS["muted"], font=("Segoe UI", 15)).pack(pady=(0, 45))
        row = tk.Frame(card, bg="white")
        row.pack()
        self.make_button(row, "🩺  Patient Prediction View", COLORS["green"], self.show_patient_form, 18).grid(row=0, column=0, padx=35, ipadx=40, ipady=25)
        self.make_button(row, "🔐  Administration Login", COLORS["blue"], self.show_login, 18).grid(row=0, column=1, padx=35, ipadx=40, ipady=25)
        tk.Label(card, text="Educational medical decision-support system — final diagnosis must be made by a qualified doctor.", bg="white", fg=COLORS["muted"], font=("Segoe UI", 11, "italic")).pack(pady=45)

    def show_login(self):
        self.clear()
        self.header("Administration Login", "Secure access for statistics, history, graphs, patient counts, and model performance", True)
        box = tk.Frame(self, bg="white", highlightbackground="#bfdbfe", highlightthickness=2)
        box.pack(pady=90, ipadx=70, ipady=50)
        tk.Label(box, text="Admin Login", bg="white", fg=COLORS["navy"], font=("Segoe UI", 25, "bold")).grid(row=0, column=0, columnspan=2, pady=25)
        user = tk.StringVar(); pwd = tk.StringVar()
        tk.Label(box, text="Username", bg="white", font=("Segoe UI", 14, "bold")).grid(row=1, column=0, sticky="e", padx=15, pady=12)
        tk.Entry(box, textvariable=user, font=("Segoe UI", 14), width=24).grid(row=1, column=1, pady=12)
        tk.Label(box, text="Password", bg="white", font=("Segoe UI", 14, "bold")).grid(row=2, column=0, sticky="e", padx=15, pady=12)
        tk.Entry(box, textvariable=pwd, show="*", font=("Segoe UI", 14), width=24).grid(row=2, column=1, pady=12)
        def check():
            if user.get() == ADMIN_USER and pwd.get() == ADMIN_PASSWORD:
                self.show_admin_dashboard()
            else:
                messagebox.showerror("Login Failed", "Invalid username or password.")
        self.make_button(box, "Login", COLORS["blue"], check).grid(row=3, column=0, columnspan=2, pady=25)

    def metric_card(self, parent, title, value, color, r, c):
        f = tk.Frame(parent, bg=color, width=165, height=88)
        f.grid(row=r, column=c, padx=8, pady=8, sticky="nsew")
        f.grid_propagate(False)
        tk.Label(f, text=str(value), bg=color, fg="white", font=("Segoe UI", 20, "bold")).pack(pady=(12, 0))
        tk.Label(f, text=title, bg=color, fg="white", font=("Segoe UI", 9, "bold")).pack()

    def show_admin_dashboard(self):
        self.clear()
        self.header("Administration Dashboard", "Dataset statistics, patient counts, graphs, model performance, and recent records", True)
        container = tk.Frame(self, bg=COLORS["bg"])
        container.pack(fill="both", expand=True, padx=22, pady=16)
        admin_canvas, outer = self.make_scrollable_area(container, COLORS["bg"])
        stats = self.manager.stats()
        colors = [COLORS["blue"], COLORS["red"], COLORS["green"], COLORS["orange"], COLORS["purple"], COLORS["cyan"], COLORS["gold"], COLORS["navy"], COLORS["pink"], COLORS["sky"]]
        top = tk.Frame(outer, bg=COLORS["bg"]); top.pack(fill="x")
        for i, (k, v) in enumerate(stats.items()):
            self.metric_card(top, k, v, colors[i % len(colors)], i // 5, i % 5)
        analytics = tk.Frame(outer, bg="white", highlightbackground="#bfdbfe", highlightthickness=2)
        analytics.pack(fill="x", pady=12)
        tk.Label(analytics, text="Visual Analytics", bg="white", fg=COLORS["navy"], font=("Segoe UI", 20, "bold")).pack(pady=(10, 0))
        canvas = tk.Canvas(analytics, height=160, bg="white", highlightthickness=0)
        canvas.pack(fill="x", padx=20, pady=8)
        self.draw_admin_charts(canvas, stats)
        table_frame = tk.Frame(outer, bg="white", highlightbackground="#bfdbfe", highlightthickness=2)
        table_frame.pack(fill="both", expand=True, pady=8)
        title_bar = tk.Frame(table_frame, bg="white"); title_bar.pack(fill="x")
        tk.Label(title_bar, text="Recent Heart Patient Dataset Records", bg="white", fg=COLORS["navy"], font=("Segoe UI", 20, "bold")).pack(side="left", padx=18, pady=12)
        self.make_button(title_bar, "📁 View Prediction History", COLORS["purple"], self.show_history_window, 11).pack(side="right", padx=20, pady=10)
        cols = FEATURES + [target_column(self.manager.df) or "target"]
        table_area = tk.Frame(table_frame, bg="white")
        table_area.pack(fill="both", expand=True, padx=18, pady=(0, 18))
        tree = ttk.Treeview(table_area, columns=cols, show="headings", height=12)
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=95, anchor="center", stretch=False)
        for _, row in self.manager.df.tail(50).iterrows():
            tree.insert("", "end", values=[row.get(c, "") for c in cols])
        ybar = ttk.Scrollbar(table_area, orient="vertical", command=tree.yview)
        xbar = ttk.Scrollbar(table_area, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
        tree.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")
        table_area.grid_rowconfigure(0, weight=1)
        table_area.grid_columnconfigure(0, weight=1)
        self.bind_mousewheel_to_canvas(admin_canvas, tree, table_frame, analytics, top)

    def draw_admin_charts(self, canvas, stats):
        canvas.delete("all")
        total = max(stats.get("Total Patients", 1), 1)
        disease = stats.get("Heart Disease", 0)
        no_disease = stats.get("No Heart Disease", 0)
        items = [("Disease", disease, COLORS["red"]), ("No Disease", no_disease, COLORS["green"]), ("High BP", stats.get("High BP Count",0), COLORS["orange"]), ("High Chol", stats.get("High Cholesterol",0), COLORS["purple"])]
        x = 80
        for label, value, color in items:
            h = int((value / total) * 105)
            canvas.create_rectangle(x, 130-h, x+70, 130, fill=color, outline="")
            canvas.create_text(x+35, 130-h-12, text=str(value), font=("Segoe UI", 11, "bold"), fill=COLORS["navy"])
            canvas.create_text(x+35, 145, text=label, font=("Segoe UI", 10, "bold"), fill=COLORS["navy"])
            x += 170
        canvas.create_text(760, 62, text=f"Model Accuracy: {stats.get('Accuracy',0)}%", font=("Segoe UI", 18, "bold"), fill=COLORS["blue"])
        canvas.create_text(760, 92, text=f"Precision {stats.get('Precision',0)}%  •  Recall {stats.get('Recall',0)}%  •  F1 {stats.get('F1 Score',0)}%", font=("Segoe UI", 12), fill=COLORS["muted"])
        canvas.create_text(760, 120, text=f"Evaluation: {getattr(self.manager, 'evaluation_note', 'Cross-validation')}", font=("Segoe UI", 10, "italic"), fill=COLORS["muted"])

    def show_patient_form(self):
        self.clear()
        self.header("Patient Heart Prediction Form", "Enter accurate medical values to generate a heart-risk report", True)
        action_bar = tk.Frame(self, bg=COLORS["card"], highlightbackground="#bfdbfe", highlightthickness=1)
        action_bar.pack(side="top", fill="x")
        button_wrap = tk.Frame(action_bar, bg=COLORS["card"])
        button_wrap.pack(pady=10)
        self.make_button(button_wrap, "🔎 Predict & Save", COLORS["green"], self.predict_patient, 12).grid(row=0, column=0, padx=7, ipadx=10)
        self.make_button(button_wrap, "📄 Download PDF", COLORS["orange"], self.download_pdf, 12).grid(row=0, column=1, padx=7, ipadx=10)
        self.make_button(button_wrap, "🧪 Load Demo", COLORS["purple"], self.load_demo_menu, 12).grid(row=0, column=2, padx=7, ipadx=10)
        self.make_button(button_wrap, "📁 History", COLORS["blue"], self.show_history_window, 12).grid(row=0, column=3, padx=7, ipadx=10)
        self.make_button(button_wrap, "🧹 Clear", COLORS["cyan"], self.clear_form, 12).grid(row=0, column=4, padx=7, ipadx=10)

        main = tk.Frame(self, bg=COLORS["bg"])
        main.pack(fill="both", expand=True, padx=24, pady=14)
        left_shell = tk.Frame(main, bg="white", highlightbackground="#93c5fd", highlightthickness=2)
        left_shell.pack(side="left", fill="both", expand=True, padx=(0, 14))
        right_panel = tk.Frame(main, bg=COLORS["bg"], width=390)
        right_panel.pack(side="right", fill="y")
        right_panel.pack_propagate(False)

        canvas = tk.Canvas(left_shell, bg="white", highlightthickness=0)
        v_scroll = ttk.Scrollbar(left_shell, orient="vertical", command=canvas.yview)
        form = tk.Frame(canvas, bg="white")
        form.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas_window = canvas.create_window((0, 0), window=form, anchor="nw")
        canvas.configure(yscrollcommand=v_scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        v_scroll.pack(side="right", fill="y")
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(canvas_window, width=event.width))

        tk.Label(form, text="Patient Information", bg="white", fg=COLORS["blue"], font=("Segoe UI", 23, "bold")).pack(pady=(18, 14))
        grid = tk.Frame(form, bg="white")
        grid.pack(padx=10, pady=(0, 12))
        self.patient_vars = {"Patient Name": tk.StringVar(), "Contact Number": tk.StringVar()}
        labels = ["Patient Name", "Contact Number"] + FEATURES
        for i, field in enumerate(labels):
            r, c = divmod(i, 2)
            label_text = DISPLAY_NAMES.get(field, field)
            tk.Label(grid, text=label_text, bg="white", fg=COLORS["navy"], font=("Segoe UI", 10, "bold")).grid(row=r, column=c * 2, sticky="e", padx=(8, 8), pady=6)
            var = self.patient_vars.setdefault(field, tk.StringVar())
            tk.Entry(grid, textvariable=var, width=23, font=("Segoe UI", 10), relief="solid", bd=1).grid(row=r, column=c * 2 + 1, padx=(0, 14), pady=6, ipady=3)
        tk.Label(form, text="This system is for educational decision support only. Final diagnosis must be made by a qualified doctor.", bg="white", fg=COLORS["muted"], wraplength=760, font=("Segoe UI", 10, "italic")).pack(pady=(0, 18))
        for child in grid.winfo_children():
            self.bind_mousewheel_to_canvas(canvas, child)
        self.bind_mousewheel_to_canvas(canvas, form, grid, left_shell)

        guide = tk.Frame(right_panel, bg="#dbeafe", height=245)
        guide.pack(fill="x")
        tk.Label(guide, text="Input Guide", bg="#dbeafe", fg=COLORS["navy"], font=("Segoe UI", 19, "bold")).pack(pady=(18, 8))
        guide_text = "\n".join([
            "• Age: 1 to 120", "• Sex: 1 Male, 0 Female", "• Chest Pain Type: 0 to 3", "• BP: 70 to 250",
            "• Cholesterol: 80 to 700", "• FBS: 1 Yes, 0 No", "• ECG: 0 to 2", "• Max Heart Rate: 60 to 250",
            "• Exercise Angina: 1 Yes, 0 No", "• ST Depression: 0 to 10", "• Slope: 0 to 2", "• Major Vessels: 0 to 4", "• Thal: 0 to 3"
        ])
        tk.Label(guide, text=guide_text, bg="#dbeafe", fg="#334155", justify="left", font=("Segoe UI", 9)).pack(padx=18, anchor="w")

        self.result_card = tk.Frame(right_panel, bg="white", highlightbackground="#bfdbfe", highlightthickness=2)
        self.result_card.pack(fill="x", pady=12)
        self.update_result_card(None)

        chart_box = tk.Frame(right_panel, bg="white", highlightbackground="#bfdbfe", highlightthickness=2)
        chart_box.pack(fill="both", expand=True)
        tk.Label(chart_box, text="Patient vs Normal Indicators", bg="white", fg=COLORS["navy"], font=("Segoe UI", 14, "bold")).pack(pady=(10, 0))
        self.chart_canvas = tk.Canvas(chart_box, height=220, bg="white", highlightthickness=0)
        self.chart_canvas.pack(fill="both", expand=True, padx=8, pady=8)
        self.draw_patient_chart(None)

    def update_result_card(self, report):
        for w in self.result_card.winfo_children():
            w.destroy()
        if not report:
            tk.Label(self.result_card, text="Result Preview", bg="white", fg=COLORS["navy"], font=("Segoe UI", 17, "bold")).pack(pady=(16, 4))
            tk.Label(self.result_card, text="Prediction result, risk percentage, alert, and explanation will appear here.", bg="white", fg=COLORS["muted"], wraplength=330, justify="center", font=("Segoe UI", 10)).pack(padx=20, pady=(0, 18))
            return
        level = report["level"]
        bg = COLORS["light_red"] if level == "High Risk" else COLORS["light_yellow"] if level == "Medium Risk" else COLORS["light_green"]
        fg = COLORS["red"] if level == "High Risk" else COLORS["gold"] if level == "Medium Risk" else COLORS["green"]
        self.result_card.configure(bg=bg)
        for child in self.result_card.winfo_children(): child.configure(bg=bg)
        tk.Label(self.result_card, text="Prediction Result", bg=bg, fg=COLORS["navy"], font=("Segoe UI", 17, "bold")).pack(pady=(14, 4))
        tk.Label(self.result_card, text=report["prediction"], bg=bg, fg=fg, font=("Segoe UI", 14, "bold"), wraplength=330).pack(pady=2)
        tk.Label(self.result_card, text=f"{report['probability']*100:.2f}% Risk Probability", bg=bg, fg=fg, font=("Segoe UI", 20, "bold")).pack(pady=4)
        tk.Label(self.result_card, text=f"⚠ {level}", bg=bg, fg=fg, font=("Segoe UI", 16, "bold")).pack(pady=(0, 8))
        tk.Label(self.result_card, text=report["explanation"], bg=bg, fg=COLORS["navy"], wraplength=330, justify="center", font=("Segoe UI", 10)).pack(padx=15, pady=(0, 14))

    def draw_patient_chart(self, values):
        c = self.chart_canvas
        if c is None:
            return
        c.delete("all")
        if not values:
            c.create_text(180, 100, text="After prediction, chart will compare patient values with healthy reference values.", width=330, fill=COLORS["muted"], font=("Segoe UI", 10), justify="center")
            return
        items = [("BP", values["trestbps"], 120), ("Chol", values["chol"], 200), ("HR", values["thalach"], 170), ("ST", values["oldpeak"], 1)]
        x = 28
        for label, val, normal in items:
            maxv = max(val, normal, 1)
            patient_h = int((val / maxv) * 100)
            normal_h = int((normal / maxv) * 100)
            c.create_rectangle(x, 130-patient_h, x+28, 130, fill=COLORS["blue"], outline="")
            c.create_rectangle(x+34, 130-normal_h, x+62, 130, fill=COLORS["green"], outline="")
            c.create_text(x+31, 148, text=label, fill=COLORS["navy"], font=("Segoe UI", 9, "bold"))
            c.create_text(x+14, 138-patient_h, text=str(int(val)), fill=COLORS["blue"], font=("Segoe UI", 8, "bold"))
            c.create_text(x+48, 138-normal_h, text=str(int(normal)), fill=COLORS["green"], font=("Segoe UI", 8, "bold"))
            x += 82
        c.create_text(90, 190, text="Blue: Patient", fill=COLORS["blue"], font=("Segoe UI", 9, "bold"))
        c.create_text(230, 190, text="Green: Normal", fill=COLORS["green"], font=("Segoe UI", 9, "bold"))

    def get_demo_samples(self):
        """Return demo cases ordered by the actual trained model probability.
        This prevents the presentation demo buttons from behaving inversely if the
        model learns slightly different boundaries from the dataset.
        """
        candidates = []
        base_cases = list(SAMPLES.values()) + [
            {"Patient Name": "High Risk Patient", "Contact Number": "03000000001", "age": 68, "sex": 1, "cp": 0, "trestbps": 190, "chol": 340, "fbs": 1, "restecg": 2, "thalach": 88, "exang": 1, "oldpeak": 6.2, "slope": 2, "ca": 4, "thal": 3},
            {"Patient Name": "High Risk Patient", "Contact Number": "03000000002", "age": 59, "sex": 1, "cp": 0, "trestbps": 175, "chol": 300, "fbs": 1, "restecg": 1, "thalach": 105, "exang": 1, "oldpeak": 4.8, "slope": 2, "ca": 3, "thal": 3},
            {"Patient Name": "Low Risk Patient", "Contact Number": "03000000003", "age": 32, "sex": 0, "cp": 3, "trestbps": 108, "chol": 155, "fbs": 0, "restecg": 0, "thalach": 188, "exang": 0, "oldpeak": 0.0, "slope": 0, "ca": 0, "thal": 1},
            {"Patient Name": "Low Risk Patient", "Contact Number": "03000000004", "age": 36, "sex": 1, "cp": 2, "trestbps": 118, "chol": 175, "fbs": 0, "restecg": 0, "thalach": 182, "exang": 0, "oldpeak": 0.1, "slope": 0, "ca": 0, "thal": 1},
            {"Patient Name": "Medium Risk Patient", "Contact Number": "03000000005", "age": 50, "sex": 1, "cp": 1, "trestbps": 138, "chol": 218, "fbs": 0, "restecg": 1, "thalach": 148, "exang": 0, "oldpeak": 1.6, "slope": 1, "ca": 1, "thal": 2},
        ]
        for case in base_cases:
            values = {k: case[k] for k in FEATURES}
            _, prob, _ = self.manager.predict(values)
            candidates.append((prob, case.copy()))
        candidates.sort(key=lambda x: x[0])
        low = candidates[0][1]
        high = candidates[-1][1]
        medium = min(candidates, key=lambda x: abs(x[0] - 0.50))[1]
        high.update({"Patient Name": "Ali Khan", "Contact Number": "03001234567"})
        medium.update({"Patient Name": "Sara Ahmed", "Contact Number": "03111234567"})
        low.update({"Patient Name": "Ahmed Raza", "Contact Number": "03221234567"})
        return {
            "High Risk Demo": high,
            "Medium Risk Demo": medium,
            "Low Risk Demo": low,
        }

    def load_demo_menu(self):
        win = tk.Toplevel(self)
        win.title("Load Demo Patient")
        win.geometry("360x260")
        win.configure(bg="white")
        tk.Label(win, text="Choose Sample Case", bg="white", fg=COLORS["navy"], font=("Segoe UI", 18, "bold")).pack(pady=20)
        for name, data in self.get_demo_samples().items():
            color = COLORS["blue"] if "Medium" in name else COLORS["red"] if "High" in name else COLORS["green"]
            self.make_button(win, name, color, lambda d=data, w=win: self.load_sample(d, w), 11).pack(pady=8, ipadx=25)

    def load_sample(self, data, win=None):
        for k, v in data.items():
            if k in self.patient_vars:
                self.patient_vars[k].set(str(v))
        if win:
            win.destroy()

    def clear_form(self):
        for var in self.patient_vars.values():
            var.set("")
        self.last_report = None
        if self.result_card:
            self.result_card.configure(bg="white")
            self.update_result_card(None)
        self.draw_patient_chart(None)

    def collect_values(self):
        name = self.patient_vars["Patient Name"].get().strip()
        contact = self.patient_vars["Contact Number"].get().strip()
        if not name:
            raise ValueError("Patient name is required.")
        if not contact:
            raise ValueError("Contact number is required.")
        if not contact.replace("+", "").replace("-", "").replace(" ", "").isdigit():
            raise ValueError("Contact number should contain digits only.")
        values = {}
        for f in FEATURES:
            raw = self.patient_vars[f].get().strip()
            if raw == "":
                raise ValueError(f"{DISPLAY_NAMES[f]} is required.")
            try:
                val = float(raw)
            except ValueError:
                raise ValueError(f"{DISPLAY_NAMES[f]} must be a number.")
            lo, hi = RANGES[f]
            if not (lo <= val <= hi):
                raise ValueError(f"{DISPLAY_NAMES[f]} must be between {lo} and {hi}.")
            if f in ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"] and int(val) != val:
                raise ValueError(f"{DISPLAY_NAMES[f]} must be a whole number.")
            values[f] = val
        return values

    def recommendations(self, values, prediction, probability):
        rec = []
        if prediction == 1 or probability >= 0.40:
            rec.append("Consult a cardiologist or qualified physician as soon as possible.")
        else:
            rec.append("Maintain routine checkups and preventive care.")
        if values["trestbps"] >= 140:
            rec.append("Reduce salt intake and monitor blood pressure regularly.")
        if values["chol"] >= 240:
            rec.append("Avoid oily/fried foods and reduce saturated fat intake.")
        if values["thalach"] < 120:
            rec.append("Discuss low heart-rate response with a medical professional.")
        if values["exang"] == 1:
            rec.append("Avoid heavy exertion until reviewed by a doctor.")
        rec.append("Exercise 30 minutes daily if approved by a doctor.")
        rec.append("Schedule a medical checkup and keep a record of BP, cholesterol, and symptoms.")
        return rec

    def explanation(self, values, prediction, probability):
        reasons = []
        if values["trestbps"] >= 140: reasons.append("high resting blood pressure")
        if values["chol"] >= 240: reasons.append("high cholesterol")
        if values["exang"] == 1: reasons.append("exercise-induced angina")
        if values["oldpeak"] >= 2: reasons.append("higher ST depression")
        if values["ca"] >= 2: reasons.append("major vessel indicator")
        if not reasons:
            reasons.append("overall values close to healthy reference ranges")
        return "The model result is influenced by " + ", ".join(reasons) + ". This is a screening result, not a final diagnosis."

    def speak(self, text):
        if not VOICE_AVAILABLE:
            return
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass

    def predict_patient(self):
        try:
            values = self.collect_values()
        except ValueError as e:
            messagebox.showwarning("Correction Required", str(e))
            return
        pred, prob, level = self.manager.predict(values)
        self.manager.append_dataset(values, pred)
        result = "Heart Disease Risk Detected" if pred == 1 else "No Heart Disease Risk Detected"
        alert = "Immediate medical attention is recommended." if level == "High Risk" else "Monitor health regularly and improve lifestyle." if level == "Medium Risk" else "Maintain a healthy lifestyle."
        self.last_report = {
            "name": self.patient_vars["Patient Name"].get().strip(),
            "contact": self.patient_vars["Contact Number"].get().strip(),
            "values": values, "prediction": result, "probability": prob, "level": level,
            "recommendations": self.recommendations(values, pred, prob),
            "explanation": self.explanation(values, pred, prob),
            "alert": alert,
            "date": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        row = {"Date": self.last_report["date"], "Patient Name": self.last_report["name"], "Contact Number": self.last_report["contact"], **values, "Prediction": result, "Risk Probability": round(prob * 100, 2), "Risk Level": level}
        pd.DataFrame([row]).to_csv(PATIENT_RECORD_FILE, mode="a", header=not os.path.exists(PATIENT_RECORD_FILE), index=False)
        self.update_result_card(self.last_report)
        self.draw_patient_chart(values)
        icon = "warning" if level in ["High Risk", "Medium Risk"] else "info"
        msg = f"Prediction: {result}\nRisk Level: {level}\nRisk Probability: {prob*100:.2f}%\n\nAlert: {alert}\n\n{self.last_report['explanation']}"
        if icon == "warning":
            messagebox.showwarning("Heart Disease Prediction Alert", msg)
        else:
            messagebox.showinfo("Heart Disease Prediction Report", msg)
        self.speak(f"Patient risk level is {level}. {alert}")

    def show_history_window(self):
        win = tk.Toplevel(self)
        win.title("Patient Prediction History")
        win.geometry("1100x520")
        win.configure(bg="white")
        tk.Label(win, text="Patient Prediction History", bg="white", fg=COLORS["navy"], font=("Segoe UI", 21, "bold")).pack(pady=14)
        if not os.path.exists(PATIENT_RECORD_FILE):
            tk.Label(win, text="No prediction history yet.", bg="white", fg=COLORS["muted"], font=("Segoe UI", 13)).pack(pady=40)
            return
        df = pd.read_csv(PATIENT_RECORD_FILE)
        cols = list(df.columns)
        tree = ttk.Treeview(win, columns=cols, show="headings", height=16)
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=110, anchor="center")
        for _, row in df.tail(100).iterrows():
            tree.insert("", "end", values=[row.get(c, "") for c in cols])
        ybar = ttk.Scrollbar(win, orient="vertical", command=tree.yview)
        xbar = ttk.Scrollbar(win, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
        tree.pack(side="left", fill="both", expand=True, padx=(14,0), pady=(0,14))
        ybar.pack(side="right", fill="y", pady=(0,14), padx=(0,14))
        xbar.pack(side="bottom", fill="x", padx=14)

    def download_pdf(self):
        if not self.last_report:
            messagebox.showwarning("No Report", "Please predict and save a patient report first.")
            return
        os.makedirs(REPORTS_DIR, exist_ok=True)
        safe_name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in self.last_report['name'].strip().replace(" ", "_")) or "patient"
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(REPORTS_DIR, f"heart_report_{safe_name}_{stamp}.pdf")
        if REPORTLAB_AVAILABLE:
            self.make_pdf(path)
            messagebox.showinfo("PDF Saved", f"Medical report saved successfully in reports folder:\n{path}")
        else:
            txt_path = path.replace(".pdf", ".txt")
            self.make_txt_report(txt_path)
            messagebox.showwarning("ReportLab Missing", f"reportlab is not installed. A text report was saved instead in reports folder:\n{txt_path}")

    def make_txt_report(self, path):
        r = self.last_report
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Heart Disease Risk Prediction Medical Report\nDate: {r['date']}\nPatient: {r['name']}\nContact: {r['contact']}\n\n")
            for k, v in r["values"].items():
                f.write(f"{DISPLAY_NAMES[k]}: {v}\n")
            f.write(f"\nPrediction: {r['prediction']}\nRisk Level: {r['level']}\nProbability: {r['probability']*100:.2f}%\nAlert: {r['alert']}\n")
            f.write("\nExplanation: " + r["explanation"] + "\n")
            f.write("\nRecommendations:\n" + "\n".join(["- " + x for x in r["recommendations"]]))

    def make_pdf(self, path):
        r = self.last_report
        doc = SimpleDocTemplate(path, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("TitleRed", parent=styles["Title"], textColor=colors.HexColor(COLORS["red"]), fontSize=22, leading=26)
        heading = ParagraphStyle("HeadingBlue", parent=styles["Heading2"], textColor=colors.HexColor(COLORS["blue"]))
        story = [Paragraph("Heart Disease Risk Prediction Medical Report", title_style), Spacer(1, 8)]
        story += [Paragraph("<b>Hospital/Clinic:</b> Smart Cardio Screening Center", styles["Normal"]), Paragraph(f"<b>Date:</b> {r['date']}", styles["Normal"]), Spacer(1, 10)]
        patient_table = Table([["Patient Name", r['name']], ["Contact Number", r['contact']], ["Prediction", r['prediction']], ["Risk Level", r['level']], ["Risk Probability", f"{r['probability']*100:.2f}%"], ["Alert", r['alert']]], colWidths=[150, 330])
        patient_table.setStyle(TableStyle([("BACKGROUND", (0,0), (0,-1), colors.HexColor("#dbeafe")), ("GRID", (0,0), (-1,-1), .5, colors.grey), ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"), ("VALIGN", (0,0), (-1,-1), "TOP")]))
        story += [patient_table, Spacer(1, 14)]
        table_data = [["Medical Attribute", "Patient Value"]] + [[DISPLAY_NAMES[k], str(v)] for k, v in r["values"].items()]
        table = Table(table_data, hAlign="LEFT", colWidths=[260, 160])
        table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor(COLORS["blue"])), ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("GRID", (0,0), (-1,-1), .5, colors.grey), ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold")]))
        story += [Paragraph("Patient Input Data", heading), table, Spacer(1, 14)]
        story += [Paragraph("Simple Explanation", heading), Paragraph(r["explanation"], styles["Normal"]), Spacer(1, 10)]
        story += [Paragraph("Lifestyle and Medical Recommendations", heading)]
        for rec in r["recommendations"]:
            story.append(Paragraph("• " + rec, styles["Normal"]))
        story += [Spacer(1, 18), Paragraph("<b>Doctor Signature:</b> ____________________________", styles["Normal"]), Spacer(1, 8), Paragraph("Disclaimer: This is an educational machine-learning screening report. It is not a final medical diagnosis. Please consult a qualified doctor.", styles["Italic"])]
        doc.build(story)


if __name__ == "__main__":
    try:
        App().mainloop()
    except Exception as exc:
        root = tk.Tk(); root.withdraw()
        messagebox.showerror("Application Error", str(exc))
