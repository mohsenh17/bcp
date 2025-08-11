from evaluation.metrics import compute_regression_metrics

class ModelEvaluator:
    def __init__(self, models, X_test, X_test_scaled, y_test):
        self.models = models
        self.X_test = X_test
        self.X_test_scaled = X_test_scaled
        self.y_test = y_test
        self.predictions = {}
        self.metrics = {}

    def evaluate(self):
        print("\n=== MODEL EVALUATION ===")
        for name, model in self.models.items():
            if name == 'Linear_Regression':
                y_pred = model.predict(self.X_test_scaled)
            else:
                y_pred = model.predict(self.X_test)

            self.predictions[name] = y_pred
            self.metrics[name] = compute_regression_metrics(self.y_test, y_pred)

            print(f"\n{name} Performance:")
            print(f"  R² Score: {self.metrics[name]['R2']:.4f}")
            print(f"  RMSE: {self.metrics[name]['RMSE']:.2f} mAh")
            print(f"  MAE: {self.metrics[name]['MAE']:.2f} mAh")
            print(f"  MSE: {self.metrics[name]['MSE']:.2f}")
        return self.metrics, self.predictions

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.config.load_config import load_config
    from src.data.data_loader import load_raw_data
    from src.data.data_preprocess import classify_capacity_bins, prepare_features_for_modeling
    from src.data.feature_engineer import process_impedance_features
    from src.models.model_factory import train_model

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
    val_size = config['training']['val_size']
    random_state = config['training']['random_state']
    prepared_data = prepare_features_for_modeling(processed_data, val_size=val_size, test_size=test_size, random_state=random_state)
    
    # Train models
    models_to_train = ["Linear_Regression", "Random_Forest"]
    trained_models = {}
    for model_name in models_to_train:
        if model_name == "Linear_Regression":
            trained_models[model_name] = train_model(model_name, prepared_data['X_train'], prepared_data['y_train'], X_train_scaled=prepared_data['X_train_scaled'])
        else:
            trained_models[model_name] = train_model(model_name, prepared_data['X_train'], prepared_data['y_train'])
    
    # Evaluate models
    evaluator = ModelEvaluator(trained_models, prepared_data['X_test'], prepared_data['X_test_scaled'], prepared_data['y_test'])
    metrics, predictions = evaluator.evaluate()