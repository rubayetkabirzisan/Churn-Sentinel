# 🛡️ Churn Sentinel
> Predictive churn management system using XGBoost + Autonomous AI Agents

## Project Structure
```
churn-sentinel/
├── data/
│   ├── raw/          ← Drop Kaggle CSV here
│   └── processed/    ← Auto-generated after preprocessing
├── notebooks/        ← EDA + model experiments
├── src/
│   ├── agents/       ← LangChain agent modules
│   ├── preprocessor.py
│   ├── model.py
│   ├── shap_explainer.py
│   └── pipeline.py
├── dashboard/        ← Streamlit app
├── outputs/          ← Saved models + email logs
├── reports/          ← Evaluation results
└── docs/             ← Final report + slides
```

## Setup Instructions

### 1. Clone & activate environment
```bash
git clone <your-repo-url>
cd churn-sentinel
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Add dataset
Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn  
Place file at: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

### 3. Run full pipeline
```bash
python src/pipeline.py
```

### 4. Launch dashboard
```bash
streamlit run dashboard/app.py
```

## Models
| Model | F1-Score | ROC-AUC |
|---|---|---|
| Logistic Regression (baseline) | TBD | TBD |
| XGBoost (improved) | TBD | TBD |

## Tech Stack
- **ML:** XGBoost, Scikit-learn, SHAP
- **Agents:** LangChain + Mistral 7B (Ollama, local, free)
- **Dashboard:** Streamlit
- **Language:** Python 3.10