"""
For different feedback sizes, check the performances.
"""
import hydra
from omegaconf import DictConfig
from src.agent.expert import Expert
from src.agent.team import ExpertTeam
from src.agent.critic import Critic
from src.utils.arranger import Arranger
from src.utils.plotmetrics import Plotter
from src.eval import Debate
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="config.yaml")
def main(config: DictConfig):
    """
    Loads the config and runs the experiment.
    """
    hydra_output_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    arranger = Arranger(config)
    feedback_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    expert_datasets, eval_data, _ = arranger.create_datasets()

    shuffled_eval_data = eval_data.shuffle()
    if config.data.name == "samsum":
        tasks = [task_set["dialogue"] for task_set in shuffled_eval_data]
        ground_truths = [task_set["summary"] for task_set in shuffled_eval_data]
    elif config.data.name == "gsm8k":
        tasks = [task_set["question"] for task_set in shuffled_eval_data]
        ground_truths = [task_set["answer"] for task_set in shuffled_eval_data]
    else:
        raise ValueError(f"Invalid dataset name: {config.dataset_name}")

    num_experts = config.experts.num_experts

    for feedback_size in feedback_sizes:
        logger.info(f"Running with feedback size: {feedback_size}")
        experts = [Expert(config, i, expert_datasets[i], eval_data) for i in range(num_experts)]
        
        for expert in experts:
            expert.feedback_size = feedback_size

        team = ExpertTeam(experts)
        critic = Critic(config)
        debate = Debate(config, team, critic)

        debate.execute_debate(tasks, ground_truths)

        plotter = Plotter(debate.metric_dict)
        plotter(hydra_output_path, f"feedback_size_{feedback_size}")

        del experts
        del team
        del critic
        del debate
        del plotter

if __name__ == "__main__":
    main()