import pandas as pd

def get_bin(capacity):
    if capacity <= 7000: return 1
    elif capacity <= 7400: return 2
    elif capacity <= 8000: return 3
    elif capacity <= 8500: return 4
    else: return 5

def classify_capacity_bins(features: pd.DataFrame) -> pd.DataFrame:
    """
    Classify samples into bins based on capacity.
    Args:
        features (pd.DataFrame): DataFrame containing capacity data.
    Returns:
        pd.DataFrame: DataFrame with capacity bins added.
    """

    
    features['Bin'] = features['Capacity (mAh)'].apply(get_bin)

    bin_stats = features.groupby('Bin').agg({
        'Capacity (mAh)': ['count', 'mean', 'std', 'min', 'max']
    }).round(2)

    print("\n=== CAPACITY CLASSIFICATION ===")
    print(bin_stats)

    return features


def prepare_features_for_modeling(processed_data, val_size=0.1, test_size=0.2, random_state=42):
    """
    Prepare features and split data into train, validation (optional), and test sets.
    Scales the training and validation/test features accordingly.

    Args:
        processed_data (pd.DataFrame): The processed dataset including features and target.
        val_size (float): Proportion of total data to use for validation. Set 0 to disable.
        test_size (float): Proportion of total data to use for test.
        random_state (int): Seed for reproducibility.

    Returns:
        dict: {
            "feature_cols": List[str],
            "X_train": pd.DataFrame,
            "X_val": pd.DataFrame or None,
            "X_test": pd.DataFrame,
            "y_train": pd.Series,
            "y_val": pd.Series or None,
            "y_test": pd.Series,
            "X_train_scaled": np.ndarray,
            "X_val_scaled": np.ndarray or None,
            "X_test_scaled": np.ndarray,
            "scaler": StandardScaler,
        }
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    feature_cols = [col for col in processed_data.columns if col not in ['Cell ID', 'Capacity (mAh)', 'Bin']]
    
    X = processed_data[feature_cols]
    y = processed_data['Capacity (mAh)']
    
    if val_size > 0:
        # First split off test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        # Then split train and val from temp
        val_relative_size = val_size / (1 - test_size)  # proportion relative to remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_relative_size, random_state=random_state
        )
    else:
        # Only train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        X_val, y_val = None, None
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val) if X_val is not None else None
    
    print(f"\n=== DATA SPLITTING ===")
    print(f"Training samples: {len(X_train)}")
    if X_val is not None:
        print(f"Validation samples: {len(X_val)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Feature columns: {len(feature_cols)}")
    
    return {
        "feature_cols": feature_cols,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from config.load_config import load_config
    
    if len(sys.argv) > 1:
        config = load_config(sys.argv[1])
    else:
        config = load_config()

    csv_file_path = config['paths']['dataset']  # Path to the battery capacity dataset

    # Load data
    data = pd.read_csv(csv_file_path)

    # Classify capacity bins
    classified_data = classify_capacity_bins(data)
    
    print("\n=== CLASSIFIED DATA ===")
    print(classified_data.head())

    # Prepare features for modeling
    prepared_data = prepare_features_for_modeling(classified_data, val_size=config['training']['val_size'], 
                                                  test_size=config['training']['test_size'],
                                                  random_state=config['training']['random_state'])  