from transformers import pipeline
from .base import BaseSummarizer

class BARTSummarizer(BaseSummarizer):
    def __init__(self):
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

    def summarize(self, texts):
        return [self.summarizer('summarize: '+text, do_sample=False)[0]["summary_text"]
                for text in texts]