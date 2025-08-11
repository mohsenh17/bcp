import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# 📊 DISTRIBUTION PLOTS
# =========================

def plot_capacity_distribution(data, save_path=None):
    """Plot histogram of battery capacities."""
    plt.figure(figsize=(8, 5))
    plt.hist(data['Capacity (mAh)'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Capacity (mAh)')
    plt.ylabel('Frequency')
    plt.title('Battery Capacity Distribution')
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_bin_distribution(processed_data, save_path=None):
    """Plot count of each capacity bin."""
    plt.figure(figsize=(6, 4))
    bin_counts = processed_data['Bin'].value_counts().sort_index()
    sns.barplot(x=bin_counts.index, y=bin_counts.values, palette='viridis')
    plt.xlabel('Capacity Bin')
    plt.ylabel('Count')
    plt.title('Battery Capacity Bin Distribution')
    for i, count in enumerate(bin_counts.values):
        plt.text(i, count, str(count), ha='center', va='bottom', fontsize=9)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# =========================
# 📈 MODEL PERFORMANCE PLOTS
# =========================

def plot_predicted_vs_actual(y_true, y_pred, model_name, save_path=None):
    """Scatter plot of predicted vs actual values."""
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             'r--', lw=2)
    plt.xlabel('Actual Capacity (mAh)')
    plt.ylabel('Predicted Capacity (mAh)')
    plt.title(f'Predicted vs Actual - {model_name}')
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_residuals(y_true, y_pred, model_name, save_path=None):
    """Plot residual errors."""
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, bins=20, kde=True, color='orange')
    plt.xlabel('Residual (mAh)')
    plt.ylabel('Frequency')
    plt.title(f'Residual Distribution - {model_name}')
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# =========================
# 📉 FEATURE IMPORTANCE
# =========================

def plot_feature_importance(model, feature_names, top_n=15, save_path=None):
    """Plot top N feature importances for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        raise ValueError(f"Model {type(model).__name__} has no feature_importances_ attribute.")

    importances = model.feature_importances_
    sorted_idx = importances.argsort()[::-1][:top_n]
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances[sorted_idx], y=[feature_names[i] for i in sorted_idx], palette="coolwarm")
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Top Feature Importances')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.config.load_config import load_config
    from src.data.data_loader import load_raw_data
    from src.data.data_preprocess import classify_capacity_bins

    # Load configuration and data
    config = load_config()
    data_path = config['paths']['dataset']
    raw_data = load_raw_data(data_path)

    # Plot capacity distribution
    plot_capacity_distribution(raw_data, save_path='capacity_distribution.png')

    # Plot bin distribution
    processed_data = classify_capacity_bins(raw_data)
    plot_bin_distribution(processed_data, save_path='bin_distribution.png')
