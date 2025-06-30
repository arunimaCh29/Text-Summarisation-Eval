import pandas as pd
import evaluate
from tqdm import tqdm

def evaluate_bertscore_scores(df, lang="en"):
    """
    Compute BERTScore (F1) for each summary column (Baseline, BART, and T5) against the document reference.
    Saves the updated DataFrame to 'data/metric_bertscore_values2.csv'.
    """
    bertscore = evaluate.load('bertscore')
    reference_col = 'document'
    summary_cols = ['summary_baseline', 'bart_summary', 't5_summary']

    all_cols = [reference_col] + summary_cols
    df = df[all_cols].copy()

    for col in summary_cols:
        df[f'{col}_bertscore'] = 0.0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating BERTScore"):
        reference = str(row[reference_col]) if pd.notnull(row[reference_col]) else ""
        for col in summary_cols:
            prediction = str(row[col]) if pd.notnull(row[col]) else ""
            try:
                if reference.strip() and prediction.strip():
                    score = bertscore.compute(predictions=[prediction], references=[reference], lang=lang)
                    df.at[idx, f'{col}_bertscore'] = score['f1'][0]
            except Exception as e:
                print(f"[{idx}] Error with {col}: {e}")
                df.at[idx, f'{col}_bertscore'] = 0.0

    df.to_csv("data/metric_bertscore_values.csv", index=False)
    print("BERTScore results saved to data/metric_bertscore_values.csv")
    return df