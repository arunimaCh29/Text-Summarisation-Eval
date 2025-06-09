from transformers import pipeline
from .base import BaseSummarizer

class T5Summarizer(BaseSummarizer):
    def __init__(self):
        self.summarizer = pipeline("summarization", model="t5-base",device =-1,num_workers = 16, batch_size= 32)

    def summarize(self, texts):
        return [self.summarizer("summarize: " + text, do_sample=False)[0]["summary_text"]
                for text in texts]
