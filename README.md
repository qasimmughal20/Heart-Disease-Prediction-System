## Heart Disease Prediction System

> An AI-powered desktop application that predicts heart disease risk using Machine Learning with **85% accuracy**.

---

## 📌 Quick Overview

This system takes 13 medical parameters as input and predicts whether a patient has heart disease.
It features a modern GUI, admin dashboard, and automated PDF report generation.

---

## ✨ Key Features

### 👨‍⚕️ Patient Portal
- Input 13 medical parameters
- Real-time prediction with risk percentage
- High/Medium/Low risk level indicator
- Personalized lifestyle recommendations
- Downloadable PDF medical reports

### 👨‍💼 Admin Dashboard
- Total patients count
- Disease vs Normal statistics
- Visual analytics cards
- One-click logout

### 📄 Automated Reports
- Professional PDF format
- Patient details + prediction
- Medical disclaimer included
- Auto-saved in `reports/` folder

### 📊 Data Analysis (Program1.py)
- Age distribution histogram
- Cholesterol outlier detection
- Chest pain type analysis
- Correlation heatmap
- ROC curve comparison
- Model accuracy comparison
- t-SNE & UMAP visualization

---

## 🚀 Installation Guide

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Step 1: Clone or Download

```bash
git clone https://github.com/YOUR_USERNAME/Heart-Disease-Prediction-System.git
cd Heart-Disease-Prediction-System
```

### Step 2: Install dependencies
```bash
pip install pandas numpy scikit-learn customtkinter reportlab matplotlib seaborn umap-learn
```

### Step 3: Run the application
```bash
python GUI.py
```


📱 How to Use

For Patients:
01. Click "PATIENT PORTAL" on main page
02. Fill all medical parameters
03. Click "PREDICT DISEASE"
04. View your risk assessment
05. Click "DOWNLOAD PDF REPORT" to save

For Admins:
01. Click "ADMINISTRATION" on main page
02. View patient statistics
03. Click "LOGOUT" to exit

🛠️ Tech Stack
Category: Technology
Language: Python 
GUI Framework: CustomTkinter
Machine Learning:	Scikit-learn
Data Processing: Pandas, NumPy
Visualization: Matplotlib, Seaborn, UMAP, t-SNE
PDF Generation: ReportLab


👨‍💻 Author
Qasim Mughal
GitHub: @qasimmughal20

LinkedIn: [Qasim Mughal](https://www.linkedin.com/in/qasimmughal-tech-dev/)

⭐ Show Support
If you found this project helpful, please give it a star ⭐ on GitHub!
