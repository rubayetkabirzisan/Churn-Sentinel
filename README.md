# 🛡️ Churn Sentinel
> An end-to-end SaaS churn prediction and autonomous retention system
> combining XGBoost + SHAP + Multi-Agent AI (LangChain + Groq)

---

## 📌 Project Overview

**Churn Sentinel** bridges the gap between churn *prediction* and
churn *prevention*. It combines:

- **XGBoost** classifier for churn probability scoring
- **SHAP** explainability for interpretable risk reasons
- **Multi-Agent AI** pipeline (LangChain + Groq) for automated
  personalized retention emails
- **Streamlit** dashboard for interactive risk visualization

---

## 🏗️ System Architecture
```
Raw Data → Preprocessing → XGBoost Scoring
    → SHAP Explanation → Behavior Detector Agent
    → Discount Agent → Email Generator Agent
    → Streamlit Dashboard + Email Log
```

---

## 📁 Project Structure
```
churn-sentinel/
├── data/
│   ├── raw/          ← Kaggle CSV here
│   └── processed/    ← Auto-generated
├── notebooks/
│   └── 01_eda.py     ← EDA + 5 plots
├── src/
│   ├── config.py          ← Central config
│   ├── preprocessor.py    ← Data pipeline
│   ├── baseline_model.py  ← Logistic Regression
│   ├── model.py           ← XGBoost + tuning
│   ├── shap_explainer.py  ← SHAP analysis
│   ├── pipeline.py        ← Full runner
│   └── agents/
│       ├── behavior_detector.py
│       ├── discount_agent.py
│       ├── email_generator.py
│       └── planner.py
├── dashboard/
│   └── app.py        ← Streamlit dashboard
├── outputs/          ← Saved models + email log
├── reports/          ← Evaluation + figures
├── docs/             ← Report + slides
├── .env.example      ← API key template
└── requirements.txt
```

---

## ⚙️ Setup Instructions

### 1. Clone repository
```bash
git clone https://github.com/YOUR_USERNAME/churn-sentinel.git
cd churn-sentinel
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add dataset
Download from Kaggle (free account required):
```
https://www.kaggle.com/datasets/blastchar/telco-customer-churn
```
Place file at:
```
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### 5. Configure API key
```bash
copy .env.example .env        # Windows
cp .env.example .env          # Mac/Linux
```
Edit `.env` and add your free Groq API key:
```
GROQ_API_KEY=gsk_your_key_here
```
Get free key at: https://console.groq.com

---

## 🚀 Running the Project

### Step 1 — Preprocess data
```bash
python src/preprocessor.py
```

### Step 2 — Train baseline model
```bash
python src/baseline_model.py
```

### Step 3 — Train XGBoost model
```bash
python src/model.py
```

### Step 4 — Run SHAP analysis
```bash
python src/shap_explainer.py
```

### Step 5 — Run agent pipeline (test mode)
```bash
python src/pipeline.py --test --size 10
```

### Step 6 — Launch dashboard
```bash
streamlit run dashboard/app.py
```

### Run full pipeline (all users)
```bash
python src/pipeline.py
```

---

## 📊 Model Results

| Metric    | Logistic Regression | XGBoost | Improvement |
|-----------|--------------------:|--------:|------------:|
| Accuracy  | ~76%  | ~80%  | +4%  |
| Precision | ~58%  | ~67%  | +9%  |
| Recall    | ~72%  | ~74%  | +2%  |
| F1-Score  | ~64%  | ~70%  | +6%  |
| ROC-AUC   | ~83%  | ~87%  | +4%  |

> Fill in exact values after training

---

## 🤖 Agent Architecture
```
Planner Agent (orchestrator)
├── SHAP Explainer    → top 3 risk reasons per user
├── Behavior Detector → disengagement vs support_issue
├── Discount Agent    → 5–10% discount eligibility
└── Email Generator   → personalized email via Groq/llama3
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | XGBoost + Scikit-learn |
| Explainability | SHAP |
| LLM | Groq API (llama-3.1-8b-instant) |
| Agent Framework | LangChain |
| Dashboard | Streamlit |
| Language | Python 3.10+ |
| Dataset | IBM Telco Churn (Kaggle) |

---

## 📚 References

1. Lemmens & Gupta (2020) — Managing Churn to Maximize Profits
2. Vafeiadis et al. (2015) — ML Techniques for Churn Prediction
3. Lundberg & Lee (2017) — SHAP: Unified Model Interpretability

