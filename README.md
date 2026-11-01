# BA-3.2-Project---TradeSmart
# TradeSmart AI

An AI solution for the automotive retail industry: it estimates the market
value of a used vehicle that a customer wants to trade in at a South African
dealership, and reads the customer's own description of the car's condition
with NLP.

Business Analysis 3.2 (AIBUY3A) project - "An AI-Powered Used-Vehicle Valuation",
Vaal University of Technology.

## Group members

| Name | Student number |
|---|---|
| LC Sekhukhune | 224946412 |
| H Madzivi | 224364065 |
| E Harmse | 240816781 |
| B Malope | 240590155 |

## What's in this repository

```
tradesmart-ml/            the AI: data, training and evaluation
  data/                   Autotrader South Africa listings (16,456 published, 16,453 after correction)
  src/
    fix_dataset.py          corrects 3 errors in the published dataset
    features.py             feature engineering shared by every model
    train_models.py         depreciation baseline, Linear Regression, Random Forest, XGBoost
    train_deep_learning.py  deep learning: multi-layer perceptron (TensorFlow/Keras)
    nlp_parser.py           NLP: TF-IDF + Logistic Regression intent classifier
    time_series_analysis.py value over time: depreciation curve and forecast
    tune_xgboost.py         hyperparameter tuning (randomized search, 5-fold CV)
    split_experiments.py    accuracy across split ratios, random shuffles and training sizes
  models/                 trained models
  outputs/                measured results (JSON) and figures

chatbot/                  the chatbot's AI logic
  ml_service.py           values a customer's car from make/model, year and mileage
  condition_nlp.py        turns the customer's condition answer into categorised findings

documentation/            project documentation and poster (PDF)
```

## Results (XGBoost)

| Measure | Result |
|---|---|
| Training / test split | 80/20: 13,162 training cars, 3,291 test cars |
| R² (average of 5 random 80/20 splits) | 0.91 (range 0.86-0.95) |
| Average error (MAE) | about R42,000, consistent across every split |
| Estimates within ±15% of the real price | about 88% |
| NLP classifier accuracy | 96% on 26 held-out examples |

## Running it

Python 3.12. From `tradesmart-ml`:

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cd src
python fix_dataset.py
python train_models.py
python train_deep_learning.py
python nlp_parser.py
python time_series_analysis.py
python split_experiments.py
```

Try the chatbot's AI logic (from `chatbot`, after `pip install -r requirements.txt`):

```
python -c "import ml_service; print(ml_service.service.estimate('Volkswagen', 'Polo Vivo Hatch 1.4 Trendline', 2019, 85000))"
python -c "import condition_nlp as c; print(c.analyser.analyse('no', 'small dent on the door, windscreen chip', 'aftermarket rims'))"
```

## Data source

Autotrader South Africa used-car listings, scraped and published by
Adewoyin, V. (2023) *used-car-price-prediction*, GitHub, MIT licence:
https://github.com/vaadewoyin/used-car-price-prediction