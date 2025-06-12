import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_rouge_distributions():
    df = pd.read_csv("data/model_summaries_with_rouge.csv")

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 14), constrained_layout=True)

    metrics = ['rouge1', 'rouge2', 'rougeL']
    titles = [
        'ROUGE-1: Unigram Overlap',
        'ROUGE-2: Bigram Overlap',
        'ROUGE-L: Longest Common Subsequence'
    ]
    models = ['summary_baseline', 'bart_summary', 't5_summary']
    labels = ['Baseline', 'BART', 'T5']

    for i, metric in enumerate(metrics):
        data = []
        means = []

        for model in models:
            col = f'{model}_{metric}'
            values = df[col].dropna() if col in df.columns else pd.Series(dtype=float)
            data.append(values)
            means.append(values.mean())

        box = axes[i].boxplot(data, patch_artist=True, widths=0.6, showfliers=False)

        for patch in box['boxes']:
            patch.set_facecolor('#d0d0d0')

        axes[i].set_title(titles[i], fontsize=14, weight='bold', pad=10)
        axes[i].set_ylabel("F1 Score", fontsize=12)
        axes[i].set_ylim(0, 1)
        axes[i].set_xticklabels(labels, fontsize=11)
        axes[i].tick_params(axis='y', labelsize=10)
        axes[i].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    fig.suptitle("ROUGE Score Distributions Across Bart and T5", fontsize=16, weight='bold')
    plt.show()


def plot_rouge_average_heatmap():
    df = pd.read_csv("data/model_summaries_with_rouge.csv")

    models = ['summary_baseline', 'bart_summary', 't5_summary']
    metrics = ['rouge1', 'rouge2', 'rougeL']
    metric_labels = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    model_labels = ['Baseline', 'BART', 'T5']

    avg_scores = {
        model: [df[f'{model}_{metric}'].mean() for metric in metrics]
        for model in models
    }
    score_df = pd.DataFrame(avg_scores, index=metric_labels)
    score_df.columns = model_labels

    plt.figure(figsize=(8, 4))
    sns.heatmap(
        score_df,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        cbar_kws={"label": "ROUGE F1 Score"},
        linewidths=0.1,
        linecolor='lightgray',
        square=True
    )
    plt.title("Average ROUGE F1 Scores", fontsize=14, pad=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11, rotation=0)
    plt.tight_layout()
    plt.show()

    return score_df.round(4)
