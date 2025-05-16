class Config:
    _instance = None

    def __init__(self):
        self.model = "bart"  # or "t5"
        self.dataset_path = "./data/MultiNews/"
        self.api_keys = {"perspective": "YOUR_API_KEY"}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance