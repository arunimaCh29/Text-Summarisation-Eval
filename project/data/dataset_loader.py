from datasets import load_dataset, enable_caching
from torch.utils.data import DataLoader
from config import Config
from tqdm.auto import tqdm
import os
from datetime import datetime

from evaluation.toxicity import ToxicityScorer

class DataSetLoader:

    def __init__(self):
        self.test_loader = None
        self.config = Config.get_instance()
        enable_caching()  # Enable caching for processed datasets
        self.toxicity_scorer = ToxicityScorer()

    def process_multi_news_batch(self, batch):
        """Process a batch of multi_news dataset examples. Toxicity scores for each document still needs to be added"""
        return {
            'document': [doc.split('|||||') if isinstance(doc, str) and '|||||' in doc else [doc] 
                        for doc in batch['document']],
            'num_documents': [len(doc.split('|||||')) if isinstance(doc, str) and '|||||' in doc else 1 
                            for doc in batch['document']],
            'summary': [summ.split('–')[1].strip() if '–' in summ else summ 
                       for summ in batch['summary']],
            'summary_length': [len(summ.split('–')[1]) if '–' in summ else len(summ) 
                             for summ in batch['summary']]
        }


    def process_reddit_batch(self, batch,model_all:bool=False):

        """Process a batch of reddit_tldr dataset examples"""
        processed_batch = {
            'document': [doc for doc in batch['content']], 
            'document_length': [len(str(content)) for content in batch['content']], 
            'summary_baseline': [summ for summ in batch['summary']],  
            'summary_length_baseline': [len(str(summ)) for summ in batch['summary']],
        }

        processed_batch['document_toxicity_detoxify'] = self.toxicity_scorer.score_detoxify(processed_batch['document'])
        processed_batch['summary_toxicity_detoxify'] = self.toxicity_scorer.score_detoxify(processed_batch['summary_baseline'])
  
        if model_all:
            processed_batch['document_toxicity_perspective'] = self.toxicity_scorer.score_perspective(processed_batch['document'])
           
            processed_batch['summary_toxicity_perspective'] = self.toxicity_scorer.score_perspective(processed_batch['summary_baseline'])
          
       


        return processed_batch


    def load(self, debug_mode=False, batch_size=32,dir_name:str='data',num_entries=100):
        """
        Load and process the dataset
        Args:
            debug_mode (bool): If True, only load a small subset of data
            batch_size (int): Batch size for DataLoader
        """
        if self.config.dataset == 'multi_news':
            print("Loading multi_news dataset...")
            dataset = load_dataset("multi_news")
            
            if debug_mode:
                print("Debug mode: Using subset of data")
                train_dataset = dataset['train'].select(range(1000))
            else:
                train_dataset = dataset['train']

            print("Converting to torch format...")
            train_dataset = train_dataset.with_format("torch")

            print("Processing dataset...")
            train_dataset = train_dataset.map(
                self.process_multi_news_batch,
                batched=True,
                batch_size=1000,  # Process 1000 examples at a time
                num_proc=4,       # Use 4 CPU cores
                desc="Processing multi_news dataset"
            )

        elif self.config.dataset == 'reddit_tldr':
            print("Loading reddit_tldr dataset...")
            dataset = load_dataset("webis/tldr-17", trust_remote_code=True)
            
            if debug_mode:
                print("Debug mode: Using subset of data")
                train_dataset = dataset['train'].shuffle().select(range(num_entries))
            else:
                train_dataset = dataset['train']

            train_dataset = train_dataset.with_format("torch")

            print("Processing dataset...")
            train_dataset = train_dataset.map(
                self.process_reddit_batch,
                batched=True,
                batch_size=batch_size,
                num_proc=os.cpu_count(),
                remove_columns=dataset['train'].column_names,
                desc="Processing reddit_tldr dataset"
            )
            if not os.path.exists(dir_name):
                os.makedirs(os.path.dirname(dir_name), exist_ok=True)
                
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            train_dataset.save_to_disk(os.path.join(dir_name, f'reddit_tldr_{timestamp}_debug_{debug_mode}.parquet'))
        

        self.test_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,        # Parallel data loading
            pin_memory=True,      # Faster data transfer to GPU
            prefetch_factor=2     # Pre-fetch batches
        )

        return train_dataset
    
    def process_full_dataset(self,batch_size:int=64,dir_name:str='data'):
        """Load the full dataset"""
        dataset = load_dataset("webis/tldr-17", trust_remote_code=True)
        train_dataset = dataset['train']
        # train_dataset = train_dataset.with_format("torch", device="cuda")
        train_dataset = train_dataset.to_iterable_dataset(num_shards=38)
        print('Processing dataset...')

        train_dataset = train_dataset.map(
            self.process_reddit_batch,
            batched=True,
            remove_columns=dataset['train'].column_names
        )

        print('Saving dataset...')

        if not os.path.exists(dir_name):
            os.makedirs(os.path.dirname(dir_name), exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        for batch_idx, batch in enumerate(train_dataset):
            print(f'Saving batch {batch_idx}...')
            batch.save_to_disk(os.path.join(dir_name, f'reddit_tldr_{timestamp}_{batch_idx}.parquet'))
        
        return train_dataset
    
    
    def provide_loader(self):
        """Return the DataLoader instance"""
        return self.test_loader

    # def top_docs_with_toxicity(self,n:int=100,toxicity_model:str='detoxify'):


    #     return self.test_loader

