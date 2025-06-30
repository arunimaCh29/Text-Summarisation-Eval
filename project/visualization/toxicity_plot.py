import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import cohen_kappa_score
import numpy as np
from scipy import stats



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
    plt.plot(df.index, df['summary_length_baseline'], label='Ground Truth Summary', linestyle='--')
    plt.plot(df.index, df['bart_summary_length'], label='BART Summary', linestyle='--')
    plt.plot(df.index, df['t5_summary_length'], label='T5 Summary', linestyle='--')

    plt.xlabel('Document Index')
    plt.ylabel('Token Length')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--')
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
    plt.grid(True, axis='y', linestyle='--')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

def plot_toxicity_transitions(df, top_n=100):

    df_top = df.iloc[:top_n]

    pairs = [
        ('document_toxicity_detoxify', 'summary_toxicity_detoxify', 'Document → Ground Truth', 'Document', 'Ground Truth'),
        ('summary_toxicity_detoxify', 'bart_summary_toxicity_detoxify', 'Ground Truth → BART', 'Ground Truth', 'BART'),
        ('summary_toxicity_detoxify', 't5_summary_toxicity_detoxify', 'Ground Truth → T5', 'Ground Truth', 'T5'),
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
        plt.grid(True, linestyle='--')
        plt.legend()
        plt.tight_layout()
        plt.show()


def binarize(series, threshold=None):
    if threshold is None:
        threshold = np.median(series)
    return (series > threshold).astype(int)

def plot_cohens_kappa(df):
    # Binarize variables
    doc_length_bin = binarize(df['document_length'].values, threshold=1000)
    doc_tox_bin = binarize(df['document_toxicity_detoxify'].values, threshold=0.5)
    sum_tox_bin = binarize(df['summary_toxicity_detoxify'].values, threshold=0.5)
    sum_length_bin = binarize(df['summary_length_baseline'].values, threshold=200)

    # Create contingency table for document length vs toxicity
    contingency = np.zeros((2, 2))
    for i in range(len(doc_length_bin)):
        contingency[doc_length_bin[i], doc_tox_bin[i]] += 1
    
    # Calculate percentages
    short_toxic = (contingency[0,1] / (contingency[0,0] + contingency[0,1])) * 100
    long_toxic = (contingency[1,1] / (contingency[1,0] + contingency[1,1])) * 100

    # Compute Cohen's Kappa
    kappa_doclen_doctox = cohen_kappa_score(doc_length_bin, doc_tox_bin)
    kappa_doclen_sumtox = cohen_kappa_score(doc_length_bin, sum_tox_bin)
    kappa_doctox_sumtox = cohen_kappa_score(doc_tox_bin, sum_tox_bin)
    kappa_doclen_sumlen = cohen_kappa_score( sum_length_bin, doc_length_bin)
    kappa_sumlen_sumtox = cohen_kappa_score( sum_length_bin, sum_tox_bin)
    kappa_sumlen_doctox = cohen_kappa_score( sum_length_bin, doc_tox_bin)
    
    # Print detailed analysis
    print("\nDetailed Analysis of Document Length vs Toxicity:")
    print(f"{'=' * 50}")
    print(f"Short documents (≤1000 tokens):")
    print(f"  - {short_toxic:.1f}% are toxic (toxicity > 0.5)")
    print(f"  - {100-short_toxic:.1f}% are non-toxic (toxicity ≤ 0.5)")
    print(f"\nLong documents (>1000 tokens):")
    print(f"  - {long_toxic:.1f}% are toxic (toxicity > 0.5)")
    print(f"  - {100-long_toxic:.1f}% are non-toxic (toxicity ≤ 0.5)")
    print(f"\nCohen's Kappa: {kappa_doclen_doctox:.3f}")
    print("Interpretation:")
    if abs(kappa_doclen_doctox) < 0.2:
        print("- No/minimal relationship between document length and toxicity")
    elif kappa_doclen_doctox > 0:
        print("- Positive relationship: longer documents tend to be more toxic")
    else:
        print("- Negative relationship: shorter documents tend to be more toxic")
    print(f"{'=' * 50}")

    # Create table data
    table_data = [
        ['Relationship', "Cohen's Kappa"],
        ['Document Length vs Document Toxicity', f'{kappa_doclen_doctox:.3f}'],
        ['Document Length vs Summary Toxicity', f'{kappa_doclen_sumtox:.3f}'],
        ['Document Toxicity vs Summary Toxicity', f'{kappa_doctox_sumtox:.3f}'],
        ['Document Length vs Summary Length', f'{kappa_doclen_sumlen:.3f}'],
        ['Summary Length vs Summary Toxicity', f'{kappa_sumlen_sumtox:.3f}'],
        ['Summary Length vs Document Toxicity', f'{kappa_sumlen_doctox:.3f}']
    ]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.axis('off')

    # Create table
    table = ax.table(
        cellText=table_data[1:],  # Data rows
        colLabels=table_data[0],  # Header row
        loc='center',
        cellLoc='center'
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)  # Make cells a bit larger
    
    # Make headers bold
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # Header row
            cell.set_text_props(weight='bold')
    
    # Add a title
    plt.title("Cohen's Kappa Values", pad=20)
    plt.tight_layout()
    plt.show()

def plot_length_toxicity_correlation(df, save_path=None):
    """
    Creates a scatter plot with regression line to show the relationship between document length and toxicity.
    
    Args:
        df: DataFrame containing document_length and document_toxicity_detoxify columns
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot
    sns.scatterplot(data=df, x='document_length', y='document_toxicity_detoxify', alpha=0.5)
    
    # Add regression line
    x = df['document_length']
    y = df['document_toxicity_detoxify']
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope * x + intercept
    plt.plot(x, line, color='red', label=f'R² = {r_value**2:.3f}\np-value = {p_value:.3e}')
    
    plt.xlabel('Document Length (tokens)')
    plt.ylabel('Toxicity Score')
    plt.title('Document Length vs Toxicity')
    plt.legend()
    plt.grid(True, linestyle='--')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
        
    # Print statistical summary
    print(f"\nCorrelation Analysis:")
    print(f"Correlation coefficient (r): {r_value:.3f}")
    print(f"R-squared (R²): {r_value**2:.3f}")
    print(f"p-value: {p_value:.3e}")
    print(f"Slope: {slope:.3e}")
    
    # Calculate mean toxicity for short vs long documents
    median_length = df['document_length'].median()
    short_docs_tox = df[df['document_length'] <= median_length]['document_toxicity_detoxify'].mean()
    long_docs_tox = df[df['document_length'] > median_length]['document_toxicity_detoxify'].mean()
    
    print(f"\nMean Toxicity Comparison:")
    print(f"Short documents (≤{median_length:.0f} tokens): {short_docs_tox:.3f}")
    print(f"Long documents (>{median_length:.0f} tokens): {long_docs_tox:.3f}")