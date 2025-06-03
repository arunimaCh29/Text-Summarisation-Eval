class Config:
    _instance = None

    def __init__(self):
        self.model = "bart"  # or "t5"
        self.detoxify_model = 'original'
        self.api_keys = {"perspective": "AIzaSyCe9upRXgqBr5JQe5OW2wcv7B3v8uTjdKw"}
        self.dataset = 'reddit_tldr'  # multi_news or reddit_tldr
        self.perspective_url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={self.api_keys['perspective']}"


    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance