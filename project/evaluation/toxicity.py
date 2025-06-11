from detoxify import Detoxify
import requests
from config import Config
import numpy as np

from ratelimit import limits, sleep_and_retry
import backoff
from tqdm import tqdm

REQUESTS_PER_MINUTE = 59
class ToxicityScorer:
    def __init__(self):
        self.config = Config.get_instance()
        
        self.model_detoxify = Detoxify(self.config.detoxify_model,device = self.config.device)


    def score_detoxify(self, texts):
        return self.model_detoxify.predict(texts)
    
    @sleep_and_retry
    @limits(calls=REQUESTS_PER_MINUTE, period=60)
    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=5)
    def call_api(self, data):
        response = requests.post(self.config.perspective_url, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            print("Error in scoring toxicity using Perspective API:", response.text)
            response.raise_for_status()
    
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

        for i, data in enumerate(tqdm(data_list)):
            try:
                response_json = self.call_api(data)
                scores[i] = response_json['attributeScores']['TOXICITY']['summaryScore']['value']
            except Exception as e:
                print(f"Failed to score text at index {i}: {e}")
                scores[i] = -1
                
        return scores if len(scores) > 1 else scores[0]

