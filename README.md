<div align="center">

# 🚚 Logistics Optimizer

### *Predicting delivery delays before they happen.*

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Model-9ACD32?style=for-the-badge)](https://lightgbm.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-NLP-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![GitHub Actions](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/features/actions)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

</div>

---

## Overview

Logistics Optimizer demonstrates an end-to-end data science pipeline — from raw data ingestion to business-facing dashboards. Built on the Brazilian e-commerce dataset Olist, it frames delivery performance as a **binary classification problem**: will this order arrive late? It showcases the integration of classical machine learning, deep learning NLP embeddings, Git versioning, CI automation, and strategic BI visualization in a single cohesive project. Every decision — from feature engineering to leakage prevention — is documented and reproducible.

---

## Key Highlights

- Distance between seller and customer is the strongest single predictor of late deliveries
- Heavy and bulky products significantly increase delay risk regardless of route
- Dropping `review_score` prevents data leakage from post-delivery customer feedback
- Chronic late sellers are statistically identifiable and consistently problematic across orders

---

## Pipeline Architecture

```mermaid
flowchart LR
    A[🌐 Kaggle API] --> B[⚙️ ETL & Sanitization]
    B --> C[🔧 Feature Engineering]
    C --> D[🤖 ML Modeling]
    D --> E[📊 Power BI Dashboard]

    style A fill:#4B8BBE,color:#fff,stroke:none
    style B fill:#306998,color:#fff,stroke:none
    style C fill:#FFD43B,color:#333,stroke:none
    style D fill:#9ACD32,color:#333,stroke:none
    style E fill:#F2C811,color:#333,stroke:none
```

| Stage | What happens |
|---|---|
| **Kaggle API** | Raw CSVs pulled programmatically and versioned |
| **ETL** | Null handling, dtype enforcement, delay target computation |
| **Feature Engineering** | Spatial, temporal, behavioral, and NLP features |
| **Modeling** | LightGBM with Optuna tuning, SHAP interpretation |
| **Dashboard** | Power BI consuming processed `.parquet` output |

---

## Challenges & How We Overcame Them

<table>
<tr>
<td width="33%" valign="top">

**⚠️ Datetime & Type Incompatibility**

LightGBM failed at training time due to residual datetime and string columns persisting after feature engineering. Added an explicit dtype filtering step in the ETL to drop non-numeric columns, and standardized all temporal features into integer representations extracted before column removal.

</td>
<td width="33%" valign="top">

**🔍 Target Leakage Detection**

`review_score` encodes post-delivery customer feedback, directly revealing whether an order arrived on time. Removing it eliminated artificial performance gains and forced the model to rely solely on pre-delivery signals — a more honest and production-valid setup.

</td>
<td width="33%" valign="top">

**📍 High-Cardinality Categorical Features**

City-level columns introduced excessive dimensionality and noise without adding predictive value. Replaced seller and customer city with a single `seller_customer_distance_km` feature — more informative, computationally leaner, and spatially grounded.

</td>
</tr>
</table>

---

## Tech Stack

<table>
  <tr>
    <td><b>Core</b></td>
    <td>Python · Pandas · NumPy · Scikit-learn · SQLite · Parquet</td>
  </tr>
  <tr>
    <td><b>ML & Tuning</b></td>
    <td>LightGBM · Optuna · SHAP</td>
  </tr>
  <tr>
    <td><b>NLP</b></td>
    <td>PyTorch · Transformers (BERT) · Hugging Face</td>
  </tr>
  <tr>
    <td><b>Geospatial</b></td>
    <td>Geopy · Spatial-temporal feature engineering</td>
  </tr>
  <tr>
    <td><b>Ingestion</b></td>
    <td>Kaggle API</td>
  </tr>
  <tr>
    <td><b>CI/CD</b></td>
    <td>GitHub Actions</td>
  </tr>
  <tr>
    <td><b>BI</b></td>
    <td>Power BI</td>
  </tr>
</table>

---

## Project Structure

```bash
logistics-optimizer/
│
├── .github/
│   └── workflows/
│       └── pipeline.yml                      # GitHub Actions CI pipeline
│
├── data/
│   ├── olist_customers_dataset.csv
│   ├── olist_geolocation_dataset.csv
│   ├── olist_order_items_dataset.csv
│   ├── olist_order_payments_dataset.csv
│   ├── olist_order_reviews_dataset.csv
│   ├── olist_orders_dataset.csv
│   ├── olist_products_dataset.csv
│   ├── olist_sellers_dataset.csv
│   └── product_category_name_translation.csv
│
├── .env
├── .gitignore
├── consolidated_logistics_base.parquet       # gitignored
├── eda_logistics.ipynb                       # Exploratory data analysis
├── logistics_data.csv                        # gitignored
├── logistics_optimizer_initialpresent.pdf    # Project presentation (PDF)
├── logistics_optimizer_v0.pptx              # Project presentation (PPTX)
├── logistics_visualization.pbix             # Power BI dashboard
├── main_etl.py                              # End-to-end ETL pipeline
├── modeling_logistics.ipynb                 # Model training and evaluation
├── processed_logistics_db.parquet          # gitignored
├── README.md
├── requirements-dev.txt
└── requirements.txt
```

---

## Key Results

| Metric | Score |
|---|---|
| **AUC-ROC** | `0.8167` |
| **F1-Score** | `0.34` |
| **Precision** | `0.22` |
| **Recall** | `0.69` |

> Model: LightGBM · Tuning: Optuna · Class imbalance: `class_weight="balanced"` + stratified split

**Top predictive features (SHAP):**
1. `seller_customer_distance_km`
2. `product_volume_cm3`
3. `km_per_estimated_day`
4. `product_weight_g`
5. `seller_late_rate`

---

## About the Author

I'm a data professional with strong proficiency in Python, SQL, machine learning, NLP, and data engineering — comfortable working across the full stack from raw ingestion to business-ready output. I bring enthusiasm for continuous learning, collaborative problem-solving, and ethical data practices to every project I take on. This repository is a reflection of how I think, build, and document real-world data solutions.

<div align="center">

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/gabriel-chaves-veras)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Gabverz)

</div>

---

<div align="center">
<sub>Built with rigor. Documented with care. Shipped with confidence.</sub>
</div>
