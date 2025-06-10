from .bart_summarizer import BARTSummarizer
from .t5_summarizer import T5Summarizer
from .base import BaseSummarizer

class SummarizerFactory(BaseSummarizer):
    def __init__(self):
        self.summarizers = {
            "bart": BARTSummarizer(),
            "t5": T5Summarizer()
        }

    def summarize(self,name, texts):
        return self.summarizers[name].summarize(texts)
    
    

# def get_summarizer(name):
#     if name == "bart":
#         return BARTSummarizer()
#     elif name == "t5":
#         return T5Summarizer()
#     else:
#         raise ValueError(f"Unknown summarizer: {name}")
