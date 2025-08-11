# I studied hardware design in my undergrad so this part of feature engineering comes from my background.
# I did not recall the details behind the impedance so I used chatGPT to referesh my memmory and 
# calculate these auxiliary features.

import numpy as np
import pandas as pd

def process_impedance_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and engineer impedance-based features.
    Args:
        data (pd.DataFrame): DataFrame containing impedance measurements.
    Returns:
        pd.DataFrame: DataFrame with extracted features."""
    
    zreal_cols = [c for c in data.columns if c.startswith('Zreal_')]
    zimag_cols = [c for c in data.columns if c.startswith('Zimag_')]

    features = pd.DataFrame({
        'Cell_ID': data['Cell ID'],
        'Capacity': data['Capacity (mAh)'],
    })

    
    zreal_data = data[zreal_cols]
    features['Zreal_mean'] = zreal_data.mean(axis=1)
    features['Zreal_std'] = zreal_data.std(axis=1)
    features['Zreal_min'] = zreal_data.min(axis=1)
    features['Zreal_max'] = zreal_data.max(axis=1)
    features['Zreal_range'] = features['Zreal_max'] - features['Zreal_min']

    zimag_data = data[zimag_cols]
    features['Zimag_mean'] = zimag_data.mean(axis=1)
    features['Zimag_std'] = zimag_data.std(axis=1)
    features['Zimag_min'] = zimag_data.min(axis=1)
    features['Zimag_max'] = zimag_data.max(axis=1)
    features['Zimag_range'] = features['Zimag_max'] - features['Zimag_min']

    # aux features
    features['Impedance_ratio'] = features['Zreal_mean'] / np.abs(features['Zimag_mean'])
    features['Total_impedance'] = np.sqrt(features['Zreal_mean']**2 + features['Zimag_mean']**2)

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

    # Process impedance features
    impedance_features = process_impedance_features(data)
    
    print("\n=== IMPEDANCE FEATURES ===")
    print(impedance_features.head())
    print("\n=== IMPEDANCE FEATURES STATISTICS ===")
    print(impedance_features.describe())