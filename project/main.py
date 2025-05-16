from config import Config
from data.multinews_loader import MultiNewsLoader
from preprocessing.cleaner import clean_texts
from summarizers.factory import get_summarizer
from evaluation.toxicity import ToxicityScorer
from evaluation.rouge import RougeScorer
from analysis.report import generate_report
from visualization.toxicity_plot import plot_toxicity_comparison

def main():
    config = Config.get_instance()
    docs, refs = MultiNewsLoader().load()

    cleaned_docs = clean_texts(docs)

    scorer = ToxicityScorer()
    pre_tox = scorer.score(cleaned_docs)

    summarizer = get_summarizer(config.model)
    summaries = summarizer.summarize(cleaned_docs)

    post_tox = scorer.score(summaries)
    rouge_scores = RougeScorer().score(summaries, refs)

    generate_report(pre_tox, post_tox, rouge_scores)
    plot_toxicity_comparison(pre_tox, post_tox)

if __name__ == "__main__":
    main()