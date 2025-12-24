# 🛡️ Churn Sentinel

**An Automated SaaS Churn Prediction and Retention System using AI Agents**

**Churn Sentinel** is a predictive analytics and automated retention system designed for SaaS businesses. Unlike traditional dashboards that only visualize past data, Churn Sentinel uses Machine Learning to predict future customer attrition and deploys autonomous AI agents to intervene immediately with personalized retention strategies.

This project was developed as part of the **CSE-404 Artificial Intelligence Sessional** at the **Military Institute of Science and Technology (MIST)**.

---

## 📖 Table of Contents

* [Background & Motivation](https://www.google.com/search?q=%23-background--motivation)
* [System Architecture](https://www.google.com/search?q=%23-system-architecture)
* [Key Features](https://www.google.com/search?q=%23-key-features)
* [Tech Stack](https://www.google.com/search?q=%23-tech-stack)
* [Development Roadmap](https://www.google.com/search?q=%23-development-roadmap)
* [Installation & Usage](https://www.google.com/search?q=%23-installation--usage)
* [Project Team](https://www.google.com/search?q=%23-project-team)
* [References](https://www.google.com/search?q=%23-references)

---

## 💡 Background & Motivation

In the subscription economy, retaining a customer is significantly cheaper than acquiring a new one. However, most retention strategies are **reactive**—businesses only realize a customer is unhappy after they have canceled.

**The Problem:**

1. **Latency:** Detection often happens too late to save the customer.
2. **Generic Outreach:** "Win-back" emails lack personalization and context.
3. **Manual Fatigue:** Managers cannot manually review logs for every user.

**The Solution:**
Churn Sentinel monitors behavior in real-time, predicts churn probability using historical patterns (XGBoost), and uses Generative AI (LLMs) to draft context-aware emails (e.g., offering technical support for users with bugs vs. discounts for price-sensitive users).

---

## 🏗 System Architecture

The project follows a modular pipeline approach:

1. **Data Ingestion & Preprocessing:** Handles activity logs, subscription data, and support tickets. Implements RFM (Recency, Frequency, Monetary) analysis.
2. **The "Brain" (Predictive Model):** An **XGBoost Classifier** that outputs a "Churn Probability Score" (0.0 to 1.0). Uses **SHAP** values to explain *why* a user is at risk.
3. **Behavior Change Detector:** A rule-based agent that flags specific anomalies (e.g., sudden drop in logins or spike in support tickets).
4. **Agentic Workflow:**
* **Planner Agent:** Decides the retention strategy.
* **Discount Agent:** Calculates financial viability of offers (5-15%).
* **Email Generator:** GPT-based agent that writes the email.


5. **Dashboard:** A Streamlit/Flask interface for monitoring and manual override.

---

## 🚀 Key Features

| Feature | Description |
| --- | --- |
| **Predictive Analytics** | Accurately identifies at-risk users before they churn using XGBoost. |
| **Explainable AI (XAI)** | Uses SHAP to tell admins *why* a user was flagged (e.g., "High Ticket Count"). |
| **Behavior Detection** | Identifies specific triggers like "Rage Clicking" or "Payment Delays". |
| **Smart Incentives** | AI determines the optimal discount (5%, 10%, or 15%) to maximize retention vs. revenue. |
| **Auto-Emailer** | Generates and sends hyper-personalized emails via SendGrid/SMTP. |
| **Conversational Admin** | A chatbot within the dashboard to query data (e.g., "Show me high-risk users from high-value tiers"). |

---

## 🛠 Tech Stack

* **Language:** Python
* **Machine Learning:** XGBoost, Scikit-learn, Pandas, NumPy
* **Explainability:** SHAP (SHapley Additive exPlanations)
* **Generative AI:** OpenAI API (GPT-4o/3.5-turbo)
* **Dashboarding:** Streamlit / Flask
* **Automation:** Python Scripts / ActivePieces
* **Email Service:** SendGrid API

---

## 📅 Development Roadmap

This project is executed over a 7-week timeline.

### Phase 1: Foundation (Weeks 1-2)

* [ ] **Week 1: Data & EDA**
* Collect/Simulate logs and subscription data.
* Clean data and perform Exploratory Data Analysis.


* [ ] **Week 2: Feature Engineering & Baseline**
* Engineer RFM features and activity trends.
* Train baseline XGBoost model.
* Evaluate ROC-AUC and Precision/Recall.



### Phase 2: Core Intelligence (Weeks 3-4)

* [ ] **Week 3: Prediction Pipeline**
* Implement `predict.py` with threshold alerts.
* Integrate Behavior Change Detector.
* Add SHAP visualizations.


* [ ] **Week 4: Generative Actions**
* Design GPT prompt templates.
* Build `email_generator.py` and integrate SendGrid.



### Phase 3: Agentic Workflow & Automation (Weeks 5-6)

* [ ] **Week 5: Advanced Agents**
* Build **Discount Recommendation Agent**.
* Build **Retention Action Planner Agent**.


* [ ] **Week 6: Orchestration & UI**
* Automate the daily "Pull → Predict → Act" pipeline.
* Build Interactive Dashboard with Chat Assistant.



### Phase 4: Polish (Week 7+)

* [ ] **Week 7: Monitoring & Docs**
* Add open-rate tracking and feedback loops.
* Finalize documentation and demo assets.



---

## 💻 Installation & Usage

### Prerequisites

* Python 3.8+
* OpenAI API Key
* SendGrid API Key (Optional for email testing)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/churn-sentinel.git
cd churn-sentinel

```


2. **Install dependencies**
```bash
pip install -r requirements.txt

```


3. **Configure Environment**
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_key_here
SENDGRID_API_KEY=your_key_here

```


4. **Run the Pipeline**
```bash
# Run the daily prediction and action loop
python automation/run_daily.py

```


5. **Launch Dashboard**
```bash
streamlit run dashboard/app.py

```



---

## 👥 Project Team

**Group: A-3** | Dept. of CSE, MIST

* **Nahiyan Ashraf Siddique** (202114072)
* **GM Rasiul Hasan** (202214022)
* **Md. Ariful Islam Khan** (202214039)
* **Md. Rubaet Kabir Zishan** (202214054)

---

## 📚 References

1. A. Lemmens and S. Gupta, "Managing Churn to Maximize Profits," *Marketing Science*, 2020.
2. T. Vafeiadis et al., "A Comparison of Machine Learning Techniques for Customer Churn Prediction," *Simulation Modelling Practice and Theory*, 2015.
3. S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," *NeurIPS*, 2017.
