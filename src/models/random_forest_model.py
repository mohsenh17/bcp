from sklearn.ensemble import RandomForestRegressor

def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from config.load_config import load_config
    from data.data_loader import load_raw_data
    from data.data_preprocess import classify_capacity_bins, prepare_features_for_modeling
    from data.feature_engineer import process_impedance_features

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        config = load_config(config_path)
    else:
        config = load_config()
    
    # Load and preprocess data
    data_path = config['paths']['dataset']
    processed_data = classify_capacity_bins(load_raw_data(data_path))
    processed_data = process_impedance_features(processed_data)

    # Split data
    test_size = config['training']['test_size']
    val_size = 0 # config['training']['val_size']
    random_state = config['training']['random_state']
    prepared_data = prepare_features_for_modeling(processed_data, val_size=val_size, test_size=test_size, random_state=random_state)
    
    
    # Train model
    model = train_random_forest(prepared_data['X_train'], prepared_data['y_train'])
    
    # Evaluate model
    train_score = model.score(prepared_data['X_train'], prepared_data['y_train'])
    test_score = model.score(prepared_data['X_test'], prepared_data['y_test'])
    print(f"Train R^2 Score: {train_score:.4f}")
    print(f"Test R^2 Score: {test_score:.4f}")