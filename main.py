"""
Starting project
"""

import hydra
from omegaconf import DictConfig
from src.agent.expert import Expert
from src.agent.critic import Critic
from src.data import Data
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="config.yaml")
def main(config: DictConfig):
    """
    Loads the config and runs the experiment.
    """
    data = Data(config)
    dataset = data.get_tokenized_data()
    train_data = dataset["train"]
    eval_data = dataset["validation"]
    test_data = dataset["test"]

    logger.info(f"Train data size: {len(train_data)}")
    logger.info(f"Eval data size: {len(eval_data)}")
    logger.info(f"Test data size: {len(test_data)}")

    expert = Expert(config, 0, train_data, eval_data)
    logger.info(f"Loaded expert 0")
    expert.fine_tune_std_lora(save=True)
    logger.info(f"Fine-tuned expert 0")

    task = "What is the capital of France?"
    logger.info(f"Task: {task}")
    logger.info(f"Expert 0 answer: {expert.generate(task)}")

if __name__ == "__main__":
    main()