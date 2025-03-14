from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
import os
import hashlib
import logging

logger = logging.getLogger(__name__)

class Data:
    def __init__(self, config: dict):
        self.config = config
        self.data_category = config.data.category
        self.model_name = config.experts.name
        self.dataset_name = config.data.name
        self.split = config.data.split
        self.device = config.experts.device
        self.data_cache_dir = config.data.data_cache_dir
        self._tokenized = False

        self._init_tokenizer()
        self.load_data()
        self.tokenize_data()

    def _init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _preprocess_data(self, examples, max_input_length=512, max_target_length=150):
        if self.dataset_name == "samsum":
            inputs = [f"Summarize this conversation:\n\n{ex}\n\n" for ex in examples['dialogue']]
            targets = [ex + self.tokenizer.eos_token for ex in examples['summary']]
            
            model_inputs = self.tokenizer(
                inputs,
                max_length=max_input_length,
                truncation=True,
                padding="max_length"
            )

            # For seq2seq models, we need to set the decoder_input_ids
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    targets,
                    max_length=max_target_length,
                    truncation=True,
                    padding="max_length"
                )["input_ids"]

            if self.config.experts.type == "causal":
                labels = [label[1:] + [-100] for label in labels]

            model_inputs["labels"] = labels
            return model_inputs
        else:
            raise ValueError(f"Invalid dataset name: {self.dataset_name}")

    def load_data(self):
        self.data = load_dataset(self.dataset_name)

    def tokenize_data(self, save=True):
        model_hash = hashlib.md5(self.model_name.encode()).hexdigest()
        dataset_hash = hashlib.md5(self.dataset_name.encode()).hexdigest()
        cache_file = os.path.join(self.data_cache_dir, f"{model_hash}_{dataset_hash}.cache")
        if os.path.exists(cache_file):
            logger.info(f"Loading tokenized data from {cache_file}")
            self.preprocessed_data = load_from_disk(cache_file)
        else:
            logger.info(f"Tokenizing data")
            self.preprocessed_data = self.data.map(self._preprocess_data, batched=True)
            if save:
                self.preprocessed_data.save_to_disk(cache_file)
        self._tokenized = True
        
    def save_tokenized_data(self):
        if not self._tokenized:
            raise ValueError("Data is not tokenized. Call tokenize_data() first.")
        
        if not os.path.exists(self.data_cache_dir):
            logger.info(f"Creating data cache directory {self.data_cache_dir}")
            os.makedirs(self.data_cache_dir)
        
        model_hash = hashlib.md5(self.model_name.encode()).hexdigest()
        dataset_hash = hashlib.md5(self.dataset_name.encode()).hexdigest()
        cache_file = os.path.join(self.data_cache_dir, f"{model_hash}_{dataset_hash}.cache")
        logger.info(f"Saving tokenized data to {cache_file}")
        self.preprocessed_data.save_to_disk(cache_file)
            
    def get_data(self):
        return self.data
    
    def get_tokenized_data(self):
        return self.preprocessed_data
    