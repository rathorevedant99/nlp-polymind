"""
Starting project
"""

import hydra
from omegaconf import DictConfig
from src.agent.expert import Expert
from src.agent.team import ExpertTeam
from src.agent.critic import Critic
from src.utils.arranger import Arranger
from src.eval import Debate
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="config.yaml")
def main(config: DictConfig):
    """
    Loads the config and runs the experiment.
    """
    arranger = Arranger(config)
    expert_datasets, eval_data, test_data = arranger.create_datasets()

    # logger.info(f"First row of expert 0: {expert_datasets[0][0]}")
    # logger.info(f"First row of expert 1: {expert_datasets[1][0]}")
    expert = Expert(config, 3, expert_datasets[0], eval_data)
    # expert.fine_tune_std_lora(save=False)
    critic = Critic(config)

    print(expert.model)
    print('--------------------------------')
    print(critic.model)

    # truncated_eval_data = eval_data.select(range(2))
    # truncated_test_data = test_data.select(range(2))


    # for task_set in truncated_eval_data:
    #     task = task_set["dialogue"]
    #     ground_truth = task_set["summary"]
    #     answer = expert.generate(task)
    #     # logger.info(f"Task: {task}")
    #     # logger.info(f"Ground truth: {ground_truth}")
    #     logger.info(f"Answer: {answer}")


if __name__ == "__main__":
    main()