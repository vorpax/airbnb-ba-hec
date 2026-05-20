# Airbnb Paris — Pricing Analytics

Streamlit application analyzing Airbnb listings in Paris to identify pricing drivers, build predictive models, and segment listings through unsupervised learning.

**Course**: Business Analytics Using Python — HEC Paris, 2026

---

## Overview

This project addresses the core question: *What are the key drivers of Airbnb listing prices in Paris, and can we build a reliable model to help hosts set an optimal price?*

The analysis covers the full pipeline — from exploratory data analysis and feature engineering, through supervised regression models, to K-Means clustering and interactive visualization.

---

## Features

- **EDA**: Price distributions, correlation heatmaps, neighborhood breakdowns, room-type comparisons
- **Geospatial Analysis**: Interactive map with pricing overlaid on Paris arrondissements; haversine distances to major landmarks
- **Amenity Analysis**: Frequency ranking, premium amenities identification, decision-tree amenity impact
- **Feature Selection**: Pearson correlation, Lasso (L1) regularization, Random Forest importance
- **Supervised Modeling**: Linear Regression, Lasso, Random Forest, XGBoost, MLP — cross-validated with grid search tuning
- **Clustering**: K-Means (elbow + silhouette selection), PCA 2D visualization, t-SNE, hierarchical dendrogram, cluster profiling
- **Temporal Analysis**: Review activity over time, seasonal decomposition

---

## Tech Stack

- Python 3.12
- Streamlit (UI)
- scikit-learn, XGBoost (ML)
- pandas, NumPy (data processing)
- Matplotlib, Seaborn (static plots)
- GeoPandas, Folium, Contextily (geospatial)
- statsmodels (seasonal decomposition)
- SHAP (model explainability)

---

## Project Structure

```
airbnb/
├── app.py                        # Streamlit entry point
├── generate_plots.py             # Batch plot generation script
├── airbnb_paris_analysis.ipynb   # Jupyter notebook (full analysis)
├── Listings.csv                  # Airbnb listings dataset (Paris subset)
├── Reviews.csv                   # Guest reviews dataset
├── Listings_data_dictionary.csv  # Field descriptions for listings
├── Reviews_data_dictionary.csv   # Field descriptions for reviews
├── plots/                        # Pre-generated plot assets
├── report/                       # LaTeX report source and PDF
├── slides/                       # LaTeX presentation source and PDF
├── requirements.txt
└── Dockerfile
```

---

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Streamlit app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### 3. Run via Docker

```bash
docker compose up
```

---

## Dataset

**Source**: [Inside Airbnb — Paris](https://insideairbnb.com/get-the-data/)  
**License**: Creative Commons CC0 (public domain)

The listings dataset contains ~74 columns covering property characteristics, location, host attributes, pricing, and review scores, filtered to Paris listings only.

Key variables:

| Variable | Description |
|---|---|
| `price` | Nightly price (target variable) |
| `room_type` | Entire home / Private room / Shared room / Hotel |
| `neighbourhood_cleansed` | Arrondissement |
| `accommodates` | Maximum number of guests |
| `amenities` | JSON list of listing amenities |
| `review_scores_rating` | Overall guest rating |
| `host_is_superhost` | Airbnb Superhost badge |

---

## Models

| Type | Models | Metrics |
|---|---|---|
| Regression | Linear, Lasso, LassoCV, Random Forest, XGBoost, MLP | R², RMSE, MAE |
| Clustering | K-Means | Inertia, Silhouette Score |

Best model: **XGBoost** (grid-search tuned), with feature importance explained via SHAP values.

---

## Key Findings

- Location (arrondissement + landmark proximity) and room type are the strongest price predictors.
- A curated set of ~15 amenities (pool, elevator, air conditioning) carries measurable price premiums.
- K-Means identifies 4–5 distinct listing segments corresponding to pricing strategies ranging from budget studio to premium central apartment.
- XGBoost outperforms linear baselines on R² and RMSE across 5-fold cross-validation.
