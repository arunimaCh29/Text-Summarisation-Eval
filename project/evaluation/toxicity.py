from detoxify import Detoxify

class ToxicityScorer:
    def __init__(self):
        self.model = Detoxify('original')

    def score(self, texts):
        return [self.model.predict(text)["toxicity"] for text in texts]
