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