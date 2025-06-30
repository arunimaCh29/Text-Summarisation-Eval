import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def plot_rouge_hist(df):
    #df = pd.read_csv("data/model_summaries_with_rouge.csv")

    if 'bart_summary_rouge_weighted_avg' not in df.columns or 't5_summary_rouge_weighted_avg' not in df.columns:
        raise ValueError("ROUGE weighted average columns not found in the DataFrame.")

    plt.figure(figsize=(10, 6))
    sns.histplot(df['bart_summary_rouge_weighted_avg'], color='orange', kde=True, bins=20,
                 label='BART', stat='density', alpha=0.6)
    sns.histplot(df['t5_summary_rouge_weighted_avg'], color='green', kde=True, bins=20,
                 label='T5', stat='density', alpha=0.6)

    plt.axvline(df['bart_summary_rouge_weighted_avg'].mean(), color='orange', linestyle='--', linewidth=1)
    plt.axvline(df['t5_summary_rouge_weighted_avg'].mean(), color='green', linestyle='--', linewidth=1)

    plt.title("Distribution of ROUGE Weighted Average Scores Across Models", fontsize=14, weight='bold')
    plt.xlabel("ROUGE Weighted Average Score", fontsize=12)
    plt.ylabel("Estimated Probability Density", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show() 



def plot_bert_score_plot(df):
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


