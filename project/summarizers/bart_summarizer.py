from transformers import pipeline, AutoTokenizer
from .base import BaseSummarizer
import numpy as np
from datasets import Dataset

class BARTSummarizer(BaseSummarizer):
    def __init__(self,device ='cpu'):
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0, num_workers = 16, batch_size= 32)
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.max_tokens = 1019 # index starts from 0 and trying to keep it within limit coz 'summarize:' prompt takes 4 more tokens
        self.overlap = 50
        self.device = device

    def summarize(self, texts):

        dataset = Dataset.from_dict({"text": texts})
        results = dataset.map(self.summarize_seq, batched=True, batch_size=16)

        return results['bart_summary']

    def summarize_seq(self, batch):
        all_summaries = []
        # print(self.summarizer.model.config)
        for i,text in enumerate(batch['text']):
            # print(f'Running for {i}')
            total_tokens = self.get_tokens_count(text)
            if total_tokens <= self.max_tokens:
                # Direct summarization
                summary = self.run_summarizer(text, max_length=total_tokens)
                all_summaries.append(summary)
            else:
                # Hierarchical summarization
                # print('Chunking strategy')
                chunked = self.chunk_text(text)
                chunk_summaries = [self.run_summarizer(chunk) for chunk in chunked]
                # print('summarising chunks')
                # Stage 3: Group chunk summaries and summarize them
                grouped_summaries = self.group_and_summarize(chunk_summaries)

                # Stage 5: Final summary
                if len(grouped_summaries) == 1:
                    all_summaries.append(grouped_summaries[0])
                else:
                    final_input = " ".join(grouped_summaries)
                    total_tokens = self.get_tokens_count(final_input)
                    if total_tokens <= self.max_tokens:
                        final_summary = self.run_summarizer(final_input, max_length=total_tokens)
                    else:
                        # Fallback to joining top-level summaries
                        final_summary = final_input
                    all_summaries.append(final_summary)

        return {"bart_summary":all_summaries}

    def run_summarizer(self, text, max_length=200, min_length=30):
        # print(self.get_tokens_count('summarize:'+text))
        return self.summarizer(
            'summarize:'+text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True 
        )[0]["summary_text"]

    def group_and_summarize(self, summaries):
        grouped = []
        current_group = []
        current_token_count = 0

        for s in summaries:
            tokens = self.get_tokens_count(s)
            if current_token_count + tokens > self.max_tokens:
                grouped_text = " ".join(current_group)
                grouped.append(self.run_summarizer(grouped_text, max_length=tokens))
                current_group = [s]
                current_token_count = tokens
            else:
                current_group.append(s)
                current_token_count += tokens

        if current_group:
            grouped_text = " ".join(current_group)
            grouped.append(self.run_summarizer(grouped_text))

        return grouped

    def get_embeddings(self, text):
        total_tokens = self.get_tokens_count(text)
        if total_tokens <= self.max_tokens:
            return self.get_sentence_embedding(text)
        else:
            return self.get_chuncked_embeddings(text).tolist()

    def get_tokens_count(self, text):
        return len(self.tokenizer(text, padding=False, truncation=False)['input_ids'])

    def get_sentence_embedding(self, text):
        embedding = self.model.encode(text).tolist()
        return embedding

    def get_chuncked_embeddings(self, text):
        chunks = self.chunk_text(text)
        embeddings = []
        for chunk in chunks:
            embedding = np.array(self.get_sentence_embedding(chunk))
            embeddings.append(embedding)
        return np.mean(embeddings, axis=0)

    def chunk_text(self, text):
        tokens = self.tokenizer(text, padding=False, truncation=False)['input_ids']
        chunks = []
        for i in range(0, len(tokens), self.max_tokens - self.overlap):
            # print(f'chunking of {i}')
            chunk = tokens[i:i + self.max_tokens]
            chunk_text = self.tokenizer.decode(chunk, skip_special_tokens=True)
            chunks.append(chunk_text)
            if len(chunk) < self.max_tokens:
                break
        return chunks
