# Logistics Optimizer – Delivery Delay Prediction with Machine Learning, NLP and Power BI

This project built an end-to-end machine learning pipeline to predict **delivery delays** for Brazilian e-commerce orders using the **Olist** public dataset. The solution estimated the **probability of delay per order**, combining **logistics**, **temporal** and **geographic** features, with an optional NLP stage using Portuguese **BERT** embeddings. The work also prepared the ground for **Power BI** dashboards driven by the model’s outputs.

---

## 1. Project Overview

The main objective of this project was to predict whether an order would be **delivered late** compared to its estimated delivery date.

The target variable was a binary label constructed from `order_delivered_customer_date` and `order_estimated_delivery_date`:

- `1` → delayed delivery  
- `0` → on-time delivery  

The models were trained to output **delay probabilities** (`P(delay=1)`) instead of only 0/1 labels. This allowed risk-based prioritization of orders, SLA monitoring and proactive customer communication.

The solution covered the full machine learning pipeline:

- Data ingestion and **ETL** on multiple Olist tables  
- **Feature engineering** with temporal, logistics, geographic and seller-level information  
- Optional **NLP** using Portuguese BERT embeddings for review texts  
- Model training with **LightGBM** and **CatBoost**  
- **Hyperparameter optimization** with **Optuna**  
- Evaluation with **ROC-AUC**, **PR-AUC**, **F1-score** and **recall** for delayed orders  
- Model interpretability using **feature importance** and **SHAP values**  
- Preparation of structured outputs for future **Power BI** dashboards

---

## 2. Data and Business Context

The project used the **Brazilian E-Commerce Public Dataset by Olist**, which contains real marketplace transactions involving multiple sellers, customers and products.

The following tables were used:

- `olist_orders_dataset.csv` – order lifecycle and timestamps  
- `olist_order_items_dataset.csv` – order items, sellers, product prices, freight value  
- `olist_order_payments_dataset.csv` – payment types and amounts  
- `olist_order_reviews_dataset.csv` – review scores and textual comments  
- `olist_products_dataset.csv` – product categories and attributes  
- `olist_sellers_dataset.csv` – seller data  
- `olist_geolocation_dataset.csv` – latitude/longitude by ZIP code  

From a business perspective, the project focused on **logistics performance** and **delay drivers**, including:

- seller reliability and historical delay behavior  
- distance between seller and customer  
- order composition (number of items, number of sellers, freight value, total value)  
- payment structure (method, installments)  
- seasonality and day-of-week effects  

The model’s outputs and insights were designed to support future **Power BI** dashboards for operational and strategic decision-making.

---

## 3. Pipeline Structure

The solution was structured as an end-to-end machine learning pipeline with the following main stages:

1. Data ingestion and **ETL** across all relevant Olist tables  
2. **Feature engineering** for temporal, logistics, geographic and seller-level features, plus optional text embeddings  
3. **Stratified** train/validation/test split to preserve class imbalance  
4. Model training using:
   - simple **baselines** (Logistic Regression, Random Forest)  
   - **gradient boosting** models (LightGBM, CatBoost)  
5. **Hyperparameter optimization** with **Optuna**  
6. Model evaluation with metrics tailored to **imbalanced classification**  
7. **Interpretability** with feature importance and **SHAP**  
8. Export of predictions and aggregated indicators structured for **Power BI** consumption

Each stage was implemented in Python, using `pandas`, `NumPy`, `scikit-learn`, LightGBM, CatBoost, Optuna and SHAP.

---

## 4. ETL and Data Preparation

The ETL step transformed raw CSV files into a single analytical dataset suitable for supervised learning.

### 4.1 Loading and Joining

All relevant tables were loaded with `pandas` and merged using keys such as:

- `order_id`  
- `customer_id`  
- `seller_id`  
- `product_id`  

A left-join strategy was applied from the order perspective to preserve all orders. This produced a consolidated dataset combining order, item, payment, review, product, seller and geolocation information.

### 4.2 Target Construction

The target label was derived from delivery dates:

- computed the difference between `order_delivered_customer_date` and `order_estimated_delivery_date`  
- orders delivered **after** the estimate were labeled as delayed (`1`)  
- orders delivered **on or before** the estimate were labeled as on-time (`0`)  

This created a clear binary classification problem for **delay prediction**.

### 4.3 Timestamp Processing

Timestamp columns were converted to `datetime` types and used to derive several lifecycle durations:

- time from order purchase to payment approval  
- time from approval to shipping  
- time from shipping to delivery  

Timestamps that could introduce **target leakage** (future information unavailable at prediction time) were identified and removed from the set of modeling features.

### 4.4 Data Cleaning and Column Selection

The ETL process handled missing values and standardized data types. Irrelevant identifiers, redundant fields and review-related columns not used in the final model were excluded.

The final cleaned dataset contained:

- engineered features relevant for delay prediction  
- no explicit leakage from future timestamps  
- structure suitable for gradient boosting models like CatBoost and LightGBM, which tolerate missing values

---

## 5. Feature Engineering

Feature engineering combined temporal, logistics, geographic, seller-level and text-based information to represent the delivery process in a rich and informative way.

### 5.1 Temporal Features

Temporal features captured process timing and calendar patterns:

- lifecycle durations:
  - purchase → approval  
  - approval → shipping  
  - shipping → delivery  

- calendar-based features:
  - day of week of the purchase  
  - month of the purchase  
  - weekend indicators and basic seasonality  

These features helped the model learn how operational lead times and calendar effects influenced delay probability.

### 5.2 Logistics and Order Composition Features

Logistics features represented the complexity and financial structure of each order:

- number of items per order  
- number of distinct sellers per order  
- total freight value  
- total order value (product prices + freight)  
- payment attributes:
  - payment method (e.g., credit card, boleto)  
  - number of installments  

These characteristics captured potential friction points in fulfillment and differences in customer payment behavior.

### 5.3 Geographic Features

Geographic features were based on seller and customer locations combined with the geolocation dataset:

- approximate **distance** between seller and customer using latitude/longitude  
- state and city indicators  
- grouping of locations into broader regions when beneficial  

These features modeled the impact of **shipping distance** and regional patterns on delivery time and delay risk.

### 5.4 Seller Performance and Aggregated Features

Seller-level aggregations captured historical performance and reliability:

- historical **delay rate** per seller  
- historical order volume per seller  
- average delivery time and variability  
- aggregated freight value and order value indicators per seller  

These features allowed the models to distinguish between reliable and high-risk sellers and were important for interpretability via SHAP.

### 5.5 Text and NLP with Portuguese BERT

The project also included an NLP component based on the reviews dataset:

- the column `review_comment_message` was processed with a **Portuguese BERT** model (BERTimbau) using `transformers` and **PyTorch**  
- sentence embeddings were generated to capture sentiment and contextual information about deliveries  
- dimensionality reduction (e.g., SVD) was applied before integrating embeddings into the tabular dataset when necessary  

This NLP step connected unstructured text (customer reviews) to the structured model features, enriching the representation of customer experience and potential issues related to delays.

---

## 6. Dataset Splitting and Validation Strategy

The dataset was split into **train**, **validation** and **test** sets using **stratified splitting** on the delay label to preserve the natural class imbalance.

- the **training set** was used to fit all candidate models  
- the **validation set** supported model comparison, hyperparameter tuning and threshold analysis  
- the **test set** served as a final hold-out sample to assess generalization performance  

This strategy provided a reliable basis for model selection and evaluation in an imbalanced classification context.

---

## 7. Modeling and Hyperparameter Optimization

The modeling stage combined simple baselines with more advanced gradient boosting methods.

### 7.1 Baseline Models

Baseline models were implemented with **scikit-learn**:

- **Logistic Regression**  
- **Random Forest**  

These models offered initial reference points for metrics such as ROC-AUC and F1-score and validated the ETL and feature engineering pipeline.

### 7.2 Main Models: LightGBM and CatBoost

The main models used for delay prediction were:

- **LightGBM**  
- **CatBoost**  

They were selected because they:

- handle missing values natively  
- work well with heterogeneous tabular data  
- provide high performance in **classification** tasks with **imbalanced data**  

Both models were trained to output delay probabilities, which supported risk ranking and flexible decision thresholds.

### 7.3 Hyperparameter Tuning with Optuna

Model performance was improved through **hyperparameter optimization** using **Optuna**.

The tuning process explored parameters such as:

- number of estimators  
- learning rate  
- maximum tree depth  
- subsample and column sample ratios  
- regularization parameters  
- class weights to mitigate class imbalance  

The best hyperparameter configurations were selected based on validation performance, mainly **ROC-AUC** and **PR-AUC**, and then evaluated on the test set.

---

## 8. Evaluation and Metrics

The evaluation considered the **imbalanced** nature of delay vs. on-time deliveries.

The main metrics used were:

- **ROC-AUC** – overall discrimination between delayed and on-time orders  
- **Precision-Recall AUC (PR-AUC)** – focus on performance in the positive (delay) class  
- **F1-score** – balance between precision and recall  
- **Recall for delayed orders** – ability to capture high-risk cases  

Additional analyses included:

- confusion matrices at different probability thresholds  
- ROC curves  
- precision-recall curves  

These analyses supported the selection of operating thresholds according to different business priorities, such as maximizing recall or balancing false positives and false negatives.

---

## 9. Model Interpretability and Business Insights

Model interpretability techniques were applied to connect machine learning outputs with actionable business insights.

The project used:

- **feature importance** from LightGBM and CatBoost  
- **SHAP values** for global and local explanations  

These tools showed how features such as seller historical delay rate, geographic distance, freight value, total order value, lifecycle durations and calendar attributes influenced the probability of delay.

The interpretability analysis revealed:

- **high-risk sellers** with consistently higher delay rates  
- **routes and regions** with increased delay probability  
- the impact of **order complexity** (multiple sellers, multiple items) on delay risk  
- time windows and seasons with higher delay incidence  

These insights were structured to feed **Power BI** dashboards, enabling interactive visualization of model outputs, historical performance and root-cause exploration.

---

## 10. Technology Stack

The project was implemented using the following tools and technologies:

- **Programming language:** Python  
- **Data manipulation and analysis:** `pandas`, `NumPy`  
- **Machine learning and statistics:** `scikit-learn`, **LightGBM**, **CatBoost`, `statsmodels`  
- **Hyperparameter optimization:** **Optuna**  
- **NLP and deep learning:** `transformers`, **PyTorch**, Portuguese **BERT** (BERTimbau)  
- **Visualization:** `matplotlib`, `seaborn`, `plotly`  
- **Business Intelligence and dashboards (prepared):** **Power BI**, DAX (using exported model outputs and aggregated indicators)  
- **Development environment and version control:** Jupyter Notebook, VS Code, Git, GitHub, `venv`  

This stack supported the complete lifecycle of the project, from data ingestion and exploratory data analysis to model training, optimization, interpretability and preparation for BI integration.
