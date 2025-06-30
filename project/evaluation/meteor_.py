import evaluate
import pandas as pd
from tqdm import tqdm

def evaluate_meteor_scores(df):
    """
    Compute METEOR scores for each summary column (BART and T5) against the document reference.
    Saves the updated DataFrame to 'data/metric_meteor_values.csv'.

    Args:
        df (pd.DataFrame): DataFrame containing original documents and model summaries.

    Returns:
        pd.DataFrame: DataFrame with METEOR scores added.
    """
    meteor = evaluate.load('meteor')
    reference_col = 'document'
    summary_cols = ['bart_summary', 't5_summary']

    retained_columns = df.columns[:4].tolist()
    df = df[retained_columns + summary_cols]

    for col in summary_cols:
        df[f'{col}_meteor'] = 0.0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating METEOR"):
        reference = row[reference_col]
        for col in summary_cols:
            prediction = row[col]
            try:
                score = meteor.compute(predictions=[prediction], references=[reference])['meteor']
                df.at[idx, f'{col}_meteor'] = score
            except:
                df.at[idx, f'{col}_meteor'] = 0.0

    df.to_csv("data/metric_meteor_values.csv", index=False)
    print(" METEOR scores saved to data/metric_meteor_values.csv")
    return df
