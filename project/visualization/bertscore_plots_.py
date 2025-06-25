import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_bertscore_distribution(df):
    """
    Plot distribution of BERTScores for Baseline, BART, and T5 summaries using KDE + Histogram.
    """
    plt.figure(figsize=(10, 5))

    sns.histplot(df['summary_baseline_bertscore'], kde=True, label="Ground-Truth", stat="density", color="orange", bins=20, alpha=0.4)
    sns.histplot(df['bart_summary_bertscore'], kde=True, label="BART", stat="density", color="purple", bins=20, alpha=0.4)
    sns.histplot(df['t5_summary_bertscore'], kde=True, label="T5", stat="density", color="skyblue", bins=20, alpha=0.4)

    plt.title("Distribution of BERTScores Across Models and Ground-Truth", fontsize=15, weight='bold')
    plt.xlabel("BERTScore (F1)", fontsize=15)
    plt.ylabel("Estimated Probability Density", fontsize=15)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Model", fontsize=11, title_fontsize=12)

    plt.tight_layout()
    plt.show()


