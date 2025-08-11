import pandas as pd

def load_raw_data(csv_file_path: str) -> pd.DataFrame:
    """
    Load raw dataset from CSV.
    Args:
        csv_file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    data = pd.read_csv(csv_file_path)
    return data

def explore_data(data: pd.DataFrame):
    """
    Print dataset overview and basic statistics.
    args:
        data (pd.DataFrame): The dataset.
    Returns:
        pd.DataFrame: Summary statistics of the dataset.
    """
    print("=== DATASET OVERVIEW ===")
    print(f"Dataset shape: {data.shape}")
    print(f"Number of battery cells: {data.shape[0]}")
    print(f"Number of features: {data.shape[1]}")

    capacity_col = 'Capacity (mAh)'
    print(f"\n=== CAPACITY STATISTICS ===")
    print(f"Mean capacity: {data[capacity_col].mean():.2f} mAh")
    print(f"Std deviation: {data[capacity_col].std():.2f} mAh")
    print(f"Min capacity: {data[capacity_col].min():.2f} mAh")
    print(f"Max capacity: {data[capacity_col].max():.2f} mAh")

    print(f"\n=== DATA QUALITY ===")
    print(f"Total missing values: {data.isnull().sum().sum()}")

    return data.describe()

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

    # Load and explore data
    data = load_raw_data(csv_file_path)
    stats = explore_data(data)
    print("\n=== SUMMARY STATISTICS ===")
    print(stats)