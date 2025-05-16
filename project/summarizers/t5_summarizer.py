from transformers import pipeline
from .base import BaseSummarizer

class T5Summarizer(BaseSummarizer):
    def __init__(self):
        self.summarizer = pipeline("summarization", model="t5-base")

    def summarize(self, texts):
        return [self.summarizer("summarize: " + text, max_length=150, min_length=40, do_sample=False)[0]["summary_text"]
                for text in texts]
