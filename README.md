# Churn Sentinel 🛡️
### An Automated SaaS Churn Prediction and Retention System using AI Agents

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Machine Learning](https://img.shields.io/badge/ML-XGBoost-orange)
![AI Agents](https://img.shields.io/badge/GenAI-OpenAI_GPT-green)
![Status](https://img.shields.io/badge/Status-Prototype-yellow)

**Course:** CSE-404 (Artificial Intelligence Sessional)  
**Institution:** Military Institute of Science and Technology (MIST)  
**Group:** A-3  

---

## 📖 Overview
**Churn Sentinel** is an end-to-end automated system designed to address customer attrition in SaaS businesses. Unlike traditional tools that only *predict* who will leave, Churn Sentinel takes proactive *action*. 

It combines **Machine Learning (XGBoost)** to identify at-risk users with **Generative AI Agents (LLMs)** to draft and send personalized retention emails. The system aims to close the loop between insight and intervention without requiring manual human oversight.

## ✨ Key Features
* **🔮 Predictive Analytics:** Uses historical activity and payment data to predict churn probability scores using **XGBoost**.
* **🧠 Explainable AI:** Integrates **SHAP (SHapley Additive exPlanations)** to explain *why* a customer is at risk (e.g., "Login frequency dropped by 50%").
* **🤖 Multi-Agent System:**
    * **Behavior Detector:** Identifies the root cause (Technical issue vs. Disengagement).
    * **Planner Agent:** Decides the retention strategy (e.g., offer discount vs. check-in).
    * **Generative Agent:** Drafts a context-aware, hyper-personalized email using OpenAI GPT models.
* **📧 Automated Intervention:** Automatically dispatches emails via **SendGrid**.
* **📊 Admin Dashboard:** A **Streamlit** interface to visualize risk scores, agent decisions, and intervention logs.

## 🛠️ Tech Stack
* **Language:** Python
* **ML Engine:** XGBoost, Scikit-learn, Pandas
* **Explainability:** SHAP
* **LLM Integration:** OpenAI API (GPT-3.5/4)
* **Email Service:** SendGrid API
* **Dashboard:** Streamlit

## 📂 Project Structure
```text
Churn-Sentinel/
├── data/                   # Raw and processed datasets
│   ├── raw_logs.csv
│   └── processed_data.csv
├── models/                 # Saved XGBoost models
│   └── churn_model.json
├── src/                    # Source code
│   ├── agents/             # AI Agent logic (Planner, Generator)
│   ├── processing/         # Data preprocessing & Feature Engineering
│   └── utils/              # Helper functions (Email sender, Config)
├── app.py                  # Streamlit Dashboard entry point
├── main.py                 # Daily automation script
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

```

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/churn-sentinel.git](https://github.com/your-username/churn-sentinel.git)
cd churn-sentinel

```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Configure Environment Variables

Create a `.env` file in the root directory and add your API keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
SENDGRID_API_KEY=your_sendgrid_api_key_here
FROM_EMAIL=your_verified_sender_email@example.com

```

## 🖥️ Usage

### Running the Dashboard

To view the analytics dashboard and manually trigger agents:

```bash
streamlit run app.py

```

### Running the Automation Pipeline

To execute the daily batch job (Ingest -> Predict -> Email):

```bash
python main.py



```

```
