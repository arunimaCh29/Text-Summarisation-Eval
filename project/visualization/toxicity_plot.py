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
        if pre_tox_summary[i]< pre_tox_document[i]:
            plt.plot([pre_length[i], pre_length[i]], [pre_tox_summary[i], pre_tox_document[i]], color='yellow', alpha=0.1)
        else:
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

def plot_document_vs_summary_lengths(df, title="Document vs Summary Lengths"):
    """
    Plots document length and summary lengths for Baseline, BART, and T5.
    """

    plt.figure(figsize=(15, 6))

    plt.plot(df.index, df['document_length'], label='Document Length', color='black', linewidth=2)
    plt.plot(df.index, df['summary_length_baseline'], label='Baseline Summary', linestyle='--')
    plt.plot(df.index, df['bart_summary_length'], label='BART Summary', linestyle='--')
    plt.plot(df.index, df['t5_summary_length'], label='T5 Summary', linestyle='--')

    plt.xlabel('Document Index')
    plt.ylabel('Token Length')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_avg_toxicity_comparison(df):
    

    toxicity_types = ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']
    sources = {
        'Document': 'document_',
        'Baseline': 'summary_',
        'BART': 'bart_summary_',
        'T5': 't5_summary_'
    }

    avg_toxicities = {source: [] for source in sources}
    for tox_type in toxicity_types:
        for label, prefix in sources.items():
            avg = df[f'{prefix}{tox_type}_detoxify'].mean()
            avg_toxicities[label].append(avg)

    bar_width = 0.2
    x = range(len(toxicity_types))
    offsets = [-1.5, -0.5, 0.5, 1.5]
    colors = ['black', '#1f77b4', '#ff7f0e', '#9467bd']

    plt.figure(figsize=(14, 6))
    for i, (label, values) in enumerate(avg_toxicities.items()):
        pos = [xi + offsets[i]*bar_width for xi in x]
        plt.bar(pos, values, width=bar_width, label=label, color=colors[i], edgecolor='black')

    plt.xticks(x, [t.title().replace("_", " ") for t in toxicity_types])
    plt.ylabel("Average Toxicity Score")
    plt.title("Toxicity Comparison: Document vs Summaries")
    plt.legend(title="Source")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

def plot_toxicity_transitions(df, top_n=100):

    df_top = df.iloc[:top_n]

    pairs = [
        ('document_toxicity_detoxify', 'summary_toxicity_detoxify', 'Document → Baseline', 'Document', 'Baseline'),
        ('summary_toxicity_detoxify', 'bart_summary_toxicity_detoxify', 'Baseline → BART', 'Baseline', 'BART'),
        ('summary_toxicity_detoxify', 't5_summary_toxicity_detoxify', 'Baseline → T5', 'Baseline', 'T5'),
        ('bart_summary_toxicity_detoxify', 't5_summary_toxicity_detoxify', 'BART → T5', 'BART', 'T5')
    ]

    for col1, col2, title, label1, label2 in pairs:
        plt.figure(figsize=(10, 4))

        for idx in df_top.index:
            y1 = df_top.loc[idx, col1]
            y2 = df_top.loc[idx, col2]
            plt.plot([idx, idx], [y1, y2], 'k:', alpha=0.5)
            plt.plot(idx, y1, 'o', color='tab:blue', label=label1 if idx == df_top.index[0] else "")
            plt.plot(idx, y2, 's', color='tab:orange', label=label2 if idx == df_top.index[0] else "")

        plt.title(f'Toxicity: {title}', fontsize=13)
        plt.xlabel('Document Index')
        plt.ylabel('Toxicity Score')
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
