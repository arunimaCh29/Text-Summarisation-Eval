import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_meteor_distribution(df):
    """
    Plot distribution of METEOR scores for BART and T5 summaries using KDE + Histogram.
    
    Args:
        csv_path (str): Path to the CSV with METEOR scores.
    """
    

    plt.figure(figsize=(10, 5))
    
    sns.histplot(df['bart_summary_meteor'], kde=True, label="BART", stat="density", color="orange", bins=20, alpha=0.4)
    sns.histplot(df['t5_summary_meteor'], kde=True, label="T5", stat="density", color="green", bins=20, alpha=0.4)

    plt.title("Distribution of METEOR Scores for BART and T5", fontsize=14, weight='bold')
    plt.xlabel("METEOR Score", fontsize=12)
    plt.ylabel("Estimated Probability Density", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
