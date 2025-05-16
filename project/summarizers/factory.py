from .bart_summarizer import BARTSummarizer
from .t5_summarizer import T5Summarizer

def get_summarizer(name):
    if name == "bart":
        return BARTSummarizer()
    elif name == "t5":
        return T5Summarizer()
    else:
        raise ValueError(f"Unknown summarizer: {name}")
