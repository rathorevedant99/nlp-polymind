"""
Starting project
"""
import hydra
from omegaconf import DictConfig
from src.agent.expert import Expert
from src.agent.team import ExpertTeam
from src.agent.critic import Critic
from src.utils.data import Data
from src.utils.arranger import Arranger
from src.utils.plotmetrics import Plotter
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

    num_experts = config.experts.num_experts
    experts = []

    for i in range(num_experts):
        expert = Expert(config, i, expert_datasets[i], eval_data)
        logger.info(f"Ready expert {i}")

        try:
            expert.fine_tune_std_lora(save=True)
            logger.info(f"Fine-tuned expert {i}")
        except Exception as e:
            logger.error(f"Error fine-tuning expert {i}: {e}")
            raise e
        experts.append(expert)   

    team = ExpertTeam(experts)

    critic = Critic(config)
    debate = Debate(config, team, critic)

    logger.info("Starting debate")

    for task_set in eval_data:
        if config.data.name == "samsum":
            tasks = [task_set["dialogue"] for task_set in eval_data]
            ground_truths = [task_set["summary"] for task_set in eval_data]
        elif config.data.name == "gsm8k":
            tasks = [task_set["question"] for task_set in eval_data]
            ground_truths = [task_set["answer"] for task_set in eval_data]
        else:
            raise ValueError(f"Invalid dataset name: {config.dataset_name}")
        
    debate.execute_debate(tasks, ground_truths)

    plotter = Plotter(debate.metric_dict)
    plotter()

    # logger.info("Evaluating first answers for unseen data")

    # for task_set in test_data:
    #     if config.data.name == "samsum":
    #         task = task_set["dialogue"]
    #         ground_truth = task_set["summary"]
    #     elif config.data.name == "gsm8k":
    #         task = task_set["question"]
    #         ground_truth = task_set["answer"]
    #     else:
    #         raise ValueError(f"Invalid dataset name: {config.dataset_name}")
        
    #     expert_answer = debate.get_final_answer(task)
    #     logger.info(f"Ground truth: {ground_truth}")
    #     logger.info(f"Expert answer: {expert_answer}")


if __name__ == "__main__":
    main()