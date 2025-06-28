from rouge_score import rouge_scorer
import pandas as pd
from tqdm import tqdm

class RougeScorer:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def score(self, summaries, references):
        return [self.scorer.score(ref, sumy) for ref, sumy in zip(references, summaries)]

# analysis/report.py
def generate_report(pre_tox, post_tox, rouge_scores):
    print("=== Toxicity Reduction ===")
    for i, (pre, post) in enumerate(zip(pre_tox, post_tox)):
        print(f"Doc {i}: Before = {pre:.3f}, After = {post:.3f}, Δ = {pre - post:.3f}")

    print("\n=== ROUGE Scores ===")
    for i, scores in enumerate(rouge_scores):
        print(f"Doc {i}: R1={scores['rouge1'].fmeasure:.3f}, R2={scores['rouge2'].fmeasure:.3f}, RL={scores['rougeL'].fmeasure:.3f}")



def evaluate_rouge_scores(df, reference_col='document', summary_cols=None):
    """
    Evaluate ROUGE-1 to ROUGE-5 scores for each summary column against the reference.
    Appends per-document ROUGE scores to the DataFrame and saves results to CSV.

    Args:
        df (pd.DataFrame): DataFrame with reference and summary columns.
        reference_col (str): Name of the reference text column.
        summary_cols (List[str]): List of summary column names.

    Returns:
        pd.DataFrame: DataFrame with added ROUGE score columns.
    """
    if summary_cols is None:
        summary_cols = ['bart_summary', 't5_summary']

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3', 'rouge4', 'rouge5'], use_stemmer=True)

    df = df.copy()

    for col in summary_cols:
        for metric in ['rouge1', 'rouge2', 'rouge3', 'rouge4', 'rouge5']:
            df[f'{col}_{metric}'] = 0.0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating ROUGE"):
        reference = row[reference_col]
        for col in summary_cols:
            summary = row[col]
            try:
                scores = scorer.score(reference, summary)
                for metric in ['rouge1', 'rouge2', 'rouge3', 'rouge4', 'rouge5']:
                    df.loc[idx, f'{col}_{metric}'] = scores[metric].fmeasure
            except:
                for metric in ['rouge1', 'rouge2', 'rouge3', 'rouge4', 'rouge5']:
                    df.loc[idx, f'{col}_{metric}'] = 0.0

    # Weighted average of ROUGE-1 to ROUGE-5
    weights = [0.3, 0.25, 0.2, 0.15, 0.1]
    for col in summary_cols:
        weighted_avg = sum(df[f'{col}_rouge{i+1}'] * weights[i] for i in range(5))
        df[f'{col}_rouge_weighted_avg'] = weighted_avg

    df.to_csv("data/metric_rouge_values.csv", index=False)
    print("Saved to data/metric_rouge_values.csv")
    return df
