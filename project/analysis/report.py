def generate_report(pre_tox, post_tox, rouge_scores):
    print("=== Toxicity Reduction ===")
    for i, (pre, post) in enumerate(zip(pre_tox, post_tox)):
        print(f"Doc {i}: Before = {pre:.3f}, After = {post:.3f}, Δ = {pre - post:.3f}")

    print("\n=== ROUGE Scores ===")
    for i, scores in enumerate(rouge_scores):
        print(f"Doc {i}: R1={scores['rouge1'].fmeasure:.3f}, R2={scores['rouge2'].fmeasure:.3f}, RL={scores['rougeL'].fmeasure:.3f}")
