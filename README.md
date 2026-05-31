# 📉 Telecom Customer Churn Analysis & Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Jupyter Notebook](https://img.shields.io/badge/Tools-Jupyter_Notebook-F37626.svg)
![Machine Learning](https://img.shields.io/badge/Model-XGBoost%20%7C%20K--Means-green.svg)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](替换你的colab连接)
[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](替换你的nbviewer连接)

## 📖 1. Project Background & Objective
In the highly competitive telecommunications industry, customer churn is the primary driver of revenue leakage. The primary objective of this project is to **reduce churn and boost overall revenue** by uncovering the hidden structural drivers of customer attrition.

By performing a deep-dive analysis on **7,043 residential phone and internet customers during Q3**, this project goes beyond surface-level metrics. It identifies critical pricing flaws, utilizes Machine Learning to segment the customer base, and predicts future churn risks, providing executive management with a data-driven roadmap for strategic adjustment.

## 🛠️ 2. Analytical Methodology
This project covers the end-to-end data science lifecycle, bridging the gap between exploratory insights and predictive deployment:

1. **Exploratory Data Analysis (EDA) & Causal Diagnostics:** Diagnosed the Q3 business landscape, focusing heavily on the performance of the newly launched promotional plan ("Offer E"). We isolated the true impact of **Fiber Optic** networks and **Value-Added Services (VAS)** on monthly charges and churn rates.
2. **Customer Segmentation (K-Means Clustering):** Applied unsupervised learning (K-Means) based on customer demographics, spending habits, and service engagement to identify distinct customer archetypes and highlight our highest-value, yet most vulnerable, segments.
3. **Churn Probability Prediction (XGBoost):** Built and optimized a highly accurate XGBoost predictive model to classify high-risk customers, enabling proactive retention campaigns before the customer decides to cancel.

## 💡 3. Key Business Insights (EDA Highlights)
Our structural drill-down revealed critical misalignments in the current pricing and promotional strategy:

* **The "Offer E" Pricing Paradox:** Despite being the newest promotional plan with a competitive baseline price, Offer E suffers from high churn. Data proves that **excessively high total Monthly Charges** post-onboarding are the primary catalysts for attrition.
* **The Fiber Optic Vulnerability:** Offer E fails to capitalize on Fiber Optic users. Surprisingly, the average monthly cost for Fiber users *without* a plan is lower than those *with* the promotional plan.
* **The VAS "Price Shock":** To eliminate confounding variables, we split the Fiber segment into "Has VAS" and "No VAS" groups. The results were clear:
  * Regardless of VAS inclusion, Offer E consistently drives the monthly fee too high.
  * The company currently lacks competitive network discounts for Fiber connections. When VAS is aggressively upsold on top of premium Fiber rates, it triggers immediate "bill shock" and subsequent churn.

## 🚀 4. Strategic Recommendations
Based on the analytical findings and model outputs, we recommend the following strategic pivots for Q4:

1. **Restructure Fiber & VAS Pricing:** For new onboarded customers (especially on Offer E), we must delay or heavily discount the upselling of Value-Added Services (VAS) during the first 6 months to prevent early lifecycle "bill shock."
2. **Implement Dedicated Fiber Retention Discounts:** Because Fiber lacks a distinct pricing advantage, introduce mandatory long-term contracts (1-2 years) coupled with meaningful, recurring network discounts to lock in high-value users.
3. **Targeted Campaigns via K-Means:** Leverage the K-Means customer profiles to allocate retention budgets efficiently. Design bespoke "Save Offers" tailored specifically to the high-value segments identified by the clustering algorithm.
4. **Proactive Intervention via XGBoost:** Deploy the XGBoost probability scores to the customer success team. Automate outreach (e.g., targeted promotional emails or courtesy calls) for the top 20% of users flagged as imminent churn risks.

## 📂 5. Repository Structure
* `data/` - Contains the raw and processed Q3 telecom datasets.
* `notebooks/` 
  * `Telcom_analyze.ipynb` - The core Jupyter Notebook containing EDA, Causal Analysis, K-Means clustering, and XGBoost modeling.
* `images/` - High-resolution outputs and executive visualizations.
  * `kmeans_customer_clusters.png` - Scatter plot of K-Means customer segments.
  * `xgb_feature_importance.png` - Top predictive drivers from the XGBoost model.
  * `xgb_roc_curve.png` - Model performance and AUC evaluation.
  * `eda_fiber_vas_churn.png` - Cross-sectional churn matrix (Fiber vs. VAS).
* `README.md` - Project documentation.
