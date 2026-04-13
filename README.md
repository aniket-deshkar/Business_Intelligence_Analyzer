# Business Intelligence Analyzer

End-to-end machine learning and analytics project built on the Olist e-commerce dataset to answer practical business questions across forecasting, customer segmentation, sentiment analysis, and operational prediction.

This repository is intended to showcase applied data science with a strong emphasis on business framing, multi-model evaluation, and artifact generation.

## Problem statement

E-commerce teams need more than dashboards. They need systems that can explain customer behavior, predict operational issues, and forecast demand. This project brings several analytics workflows into one coherent notebook-based solution.

## Core modules

- Sales prediction by state
- Late delivery classification
- Payment method prediction
- Customer segmentation with RFM and clustering
- Sentiment analysis on customer reviews
- Time-series forecasting for order volume

## Outputs and visuals

The repository includes generated visual assets and model artifacts for the main workflows:

- `regression_results.png`
- `classification_late_delivery.png`
- `classification_payment.png`
- `rfm_3d_clusters.png`
- `sentiment_distribution.png`
- `timeseries_forecast.png`
- `extended_forecast.png`
- models stored under `artifacts/`

## Main files

```text
intelligent_business_analyzer_and_forecasting_system.ipynb  Primary analysis notebook
ml_business_analytics.ipynb                                 Generated notebook variant
create_notebook.py                                          Notebook generation helper
artifacts/                                                  Saved trained models
report_assets/                                              Exported notebook images
```

## Techniques used

- Data cleaning and relational dataset merging
- Feature engineering for business and operational metrics
- Regression and classification with scikit-learn
- Customer segmentation with KMeans
- NLP with TF-IDF and sentiment analysis
- Time-series forecasting with ARIMA, SARIMA, and Prophet
- Cross-validation and comparative model evaluation

## What this demonstrates

- Ability to translate business questions into ML tasks
- Breadth across supervised, unsupervised, NLP, and time-series work
- Structured evaluation instead of training a single model in isolation
- Familiarity with artifact persistence and result communication
- Practical use of visualizations to explain outcomes

## Running the project

Install the main dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels nltk prophet
```

Optional dependency for additional NLP experimentation:

```bash
pip install transformers
```

Then open and run `intelligent_business_analyzer_and_forecasting_system.ipynb` from top to bottom.

## Why this is portfolio-relevant

For product companies, this project helps signal that you can:

- work with messy real-world relational data
- evaluate multiple modeling approaches instead of chasing one score
- connect technical work to business use cases
- present results clearly with plots, artifacts, and summaries

## Suggested next improvements

- Convert the notebook into a package or API-backed analytics service
- Add a reproducible environment file for easier setup
- Document benchmark results and final model choices in a shorter executive summary
- Add automated checks for notebook execution and data availability
