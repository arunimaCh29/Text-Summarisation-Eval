from transformers import pipeline
from .base import BaseSummarizer

class BARTSummarizer(BaseSummarizer):
    def __init__(self):
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    def summarize(self, texts):
        return [self.summarizer(text, max_length=150, min_length=40, do_sample=False)[0]["summary_text"]
                for text in texts]