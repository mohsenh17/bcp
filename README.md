# 🔋 Battery Capacity Prediction

## 📌 Overview
This project predicts **battery capacity** based on **impedance data** and other relevant features.  
It includes:
- Data loading and exploration
- Feature engineering for impedance characteristics
- Classification into capacity bins
- Flexible train/validation/test splitting
- Modular regression model training and evaluation

The repository is designed for **scalability**, **modularity**, and **reproducibility**, making it easy to integrate new models and extend the ML pipeline.

---

## 📂 Project Structure
```bash
.
├── README.md
├── configs
│   └── config.yaml             # Configuration for data paths & parameters
├── data
│   └── interview-dataset.csv   # Raw dataset
├── notebook
│   ├── 01_data_exploration.ipynb # Data exploration and visualization
│   └── README.md               # Notebook-specific notes
└── src
    ├── config
    │   └── load_config.py       # Config file loader utility
    ├── data
    │   ├── data_loader.py       # Load dataset & basic exploration
    │   ├── data_preprocess.py   # Data splitting, binning, scaling
    │   └── feature_engineer.py  # Impedance feature extraction & engineering
    └── models
        ├── __init__.py
        ├── linear_regression_model.py # Linear Regression implementation
        ├── random_forest_model.py     # Random Forest Regressor implementation
        ├── neural_network_model.py    # (Planned) Neural Network model
        ├── xgboost_model.py           # (Planned) XGBoost model
        └── model_factory.py           # Factory for loading and training models
