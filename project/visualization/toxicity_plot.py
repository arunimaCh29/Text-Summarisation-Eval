import matplotlib.pyplot as plt
import seaborn as sns


def plot_toxicity_comparison(pre_tox, post_tox, save_path=None):
    indices = list(range(len(pre_tox)))

    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    for i in indices:
        plt.plot([0, 1], [pre_tox[i], post_tox[i]], color='gray', alpha=0.6)
        plt.scatter([0], [pre_tox[i]], color='red')
        plt.scatter([1], [post_tox[i]], color='green')

    plt.xticks([0, 1], ['Original', 'Summary'], fontsize=12)
    plt.ylabel("Toxicity Score", fontsize=14)
    plt.title("Toxicity Before vs After Summarization", fontsize=16)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
