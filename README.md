# 🔋 Battery Capacity Prediction

## 📌 Overview
This project predicts **battery capacity** based on **impedance data** and other relevant features.  
It includes:
- Data loading and exploration
- Feature engineering for impedance characteristics
- Classification into capacity bins
- Model training and evaluation (planned)

The repository is structured for scalability, modularity, and reproducibility.

---

## 📂 Project Structure
```bash
.
├── README.md
├── configs
│   └── config.yaml # Configuration for data paths & parameters
├── data
│   └── interview-dataset.csv
├── notebook
│   ├── 01_data_exploration.ipynb # Data exploration and visualization
│   └── README.md
└── src
    ├── config
    │   └── load_config.py # Config file loader utility
    └── data
        ├── data_loader.py  # Load dataset & exploration
        ├── data_preprocess.py # Bining
        └── feature_engineer.py # Aux feature
```