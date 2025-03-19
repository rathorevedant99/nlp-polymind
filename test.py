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
from src.metrics import Metrics
from src.utils.plotmetrics import Plotter
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
    expert0 = Expert(config, 0, expert_datasets[0], eval_data)
    expert1 = Expert(config, 1, expert_datasets[1], eval_data)
    metrics = Metrics()
    metrics_dict = {}

    expert_answers = {}
    
    for task in eval_data:
        dialogue = task["dialogue"]
        summary = task["summary"]
        for i in range(2):
            answer0 = expert0.generate(dialogue)
            answer1 = expert1.generate(dialogue)
            expert_answers[0] = answer0
            expert_answers[1] = answer1
            rouge_scores, bertscore_scores, novelty_scores, length_ratios = metrics(summary, expert_answers)
            metrics_dict[i] = {
                "rouge_scores": rouge_scores,
                "bertscore_scores": bertscore_scores,
                "novelty_scores": novelty_scores,
                "length_ratios": length_ratios
            }
    logger.info(f"Metrics: {metrics_dict}")

    plotter = Plotter(metrics_dict)
    plotter()



if __name__ == "__main__":
    main()