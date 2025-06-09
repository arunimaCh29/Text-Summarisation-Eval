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


def plot_toxicity_comparison_with_length(pre_tox_summary, pre_tox_document, pre_length, save_path=None):
    #indices = list(range(len(pre_tox)))

    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    for i in pre_length.keys():
        plt.plot([pre_length[i], pre_length[i]], [pre_tox_summary[i], pre_tox_document[i]], color='green', alpha=0.2)
        plt.scatter([pre_length[i]], [pre_tox_summary[i]], color='blue')
        plt.scatter([pre_length[i]], [pre_tox_document[i]], color='red')

    plt.xlabel("Document Length")
    plt.ylabel("Toxicity")
    plt.legend(['Toxicity pair','Summary Toxicity', 'Document Toxicity'])
    plt.title("Document Length vs Toxicity Reported", fontsize=16)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)


def plot_toxicity_distribution(tox,save_path=None, label='Distribution of Document Toxicity', x_label ='Toxicity Value'):
    plt.figure(figsize=(10, 6))
    sns.histplot(tox, bins=30, kde=True)
    plt.title(label)
    plt.xlabel(x_label)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


