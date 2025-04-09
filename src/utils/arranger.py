"""
Author: Vedant S Rathore
Create datasets for multiple agents
"""

from src.utils.data import Data

class Arranger:
    def __init__(self, config):
        self.config = config
        self.data = Data(config)
        self.mode = config.mode
        self.num_experts = config.experts.num_experts

    def create_datasets(self):
        """
        Create datasets for multiple agents
        Returns:
            expert_datasets (List[Dataset]): List of training datasets for each expert
            eval_data (Dataset): Dataset for evaluation
            test_data (Dataset): Dataset for testing
        """
        dataset = self.data.get_tokenized_data()
        if self.config.data.name == "samsum":
            train_data = dataset["train"]
            eval_data = dataset["validation"]
            test_data = dataset["test"]
        elif self.config.data.name == "gsm8k":
            train_data = dataset["train"].train_test_split(test_size=0.1)
            train_data = train_data["train"]
            eval_data = dataset["test"]
            test_data = dataset["test"]
        elif self.config.data.name == "opus":
            train_data = dataset["train"].train_test_split(test_size=0.1)
            eval_data = train_data["test"]
            train_data = train_data["train"]
            test_data = dataset["validation"]
        else:
            raise ValueError(f"Invalid dataset name: {self.config.dataset_name}")
        
        shuffled_data = train_data.shuffle()
        dataset_size = len(shuffled_data)
        expert_dataset_size = dataset_size // self.num_experts

        test_data = test_data.shuffle(seed=42)
        expert_datasets = []

        for i in range(self.num_experts):
            start_idx = i * expert_dataset_size
            end_idx = start_idx + expert_dataset_size if i < self.num_experts - 1 else dataset_size
            expert_dataset = shuffled_data.select(range(start_idx, end_idx))
            expert_datasets.append(expert_dataset)

        if self.mode == "dev":
            eval_data = eval_data.select(range(5))
            test_data = test_data.select(range(5))

        return expert_datasets, eval_data, test_data
