
import pandas as pd
from summarizers.factory import get_summarizer

# List of input texts
texts = [
    "Consuming a healthy diet throughout the life-course helps to prevent malnutrition in all its forms as well as a range of noncommunicable diseases (NCDs) and conditions. However, increased production of processed foods, rapid urbanization and changing lifestyles have led to a shift in dietary patterns. People are now consuming more foods high in energy, fats, free sugars and salt/sodium, and many people do not eat enough fruit, vegetables and other dietary fibre such as whole grains.",
    "The subreddit used to be great, but now it's filled with trolls and off-topic posts.",
]

# List to collect all results
results = []

# Loop through models and generate summaries
for model_name in ["bart", "t5"]:
    summarizer = get_summarizer(model_name)
    summaries = summarizer.summarize(texts)

    print(f"\n===== Summaries by {model_name.upper()} =====\n")

    for i, (text, summary) in enumerate(zip(texts, summaries)):
        print(f"[{model_name.upper()}] Input {i+1}:\n{text}")
        print(f"[{model_name.upper()}] Summary {i+1}:\n{summary}\n{'-'*60}")
        results.append({
            "model": model_name.upper(),
            "input_text": text,
            "summary": summary
        })

# Save to CSV
df = pd.DataFrame(results)
df.to_csv("model_summaries_bart_t5.csv", index=False)
#print("\nAll summaries saved to model_summaries_bart_t5.csv")


