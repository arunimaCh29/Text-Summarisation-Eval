from detoxify import Detoxify
import requests
from config import Config
import numpy as np

class ToxicityScorer:
    def __init__(self):
        self.config = Config.get_instance()
        
        self.model_detoxify = Detoxify(self.config.detoxify_model,device ='cuda')


    def score_detoxify(self, texts):
        return self.model_detoxify.predict(texts)
    
    def score_perspective(self,texts):
        if isinstance(texts, str):
            texts = [texts]
       
        scores = np.zeros(len(texts))
        data_list = [
            {
                "comment": {"text": text},
                "languages": ["en"],
                "requestedAttributes": {"TOXICITY": {}}
            }
            for text in texts
        ]

        for i, data in enumerate(data_list):
            response = requests.post(self.config.perspective_url, json=data)
            response_json = response.json()
            if response.status_code == 200:
                scores[i] = response_json['attributeScores']['TOXICITY']['summaryScore']['value']
            else:
                print('Error in scoring toxicity using perspective',response_json)

        return scores if len(scores) > 1 else scores[0]

