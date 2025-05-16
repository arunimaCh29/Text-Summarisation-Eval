from rouge_score import rouge_scorer

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
