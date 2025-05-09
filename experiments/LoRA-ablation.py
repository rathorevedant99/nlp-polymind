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
from src.utils.plot_exp import plot_expert_run_performance, plot_expert_summary
import logging
import pandas as pd
from tqdm import tqdm
from time import time


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def expert_test_evaluation(team, test_tasks, test_ground_truths, metrics):
    expert_answers = {}
    expert_scores = {}

    for i, task in enumerate(test_tasks):
        expert_answers[i] = team.get_expert_answers(task)
        rouge_score = metrics.eval_rouge(test_ground_truths[i], expert_answers[i])

        exp_scores = []
        for expert_idx in range(len(team.experts)):
            exp_scores.append(rouge_score[str(expert_idx)]["rouge1"].fmeasure)
        expert_scores[i] = exp_scores

    return expert_scores
        

@hydra.main(version_base=None, config_path="configs", config_name="config.yaml")
def main(config: DictConfig):
    """
    Loads the config and runs the experiment.
    """
    runs = 10

    data = pd.DataFrame(columns=["run", "expert_id", "before", "after"])

    hydra_output_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    for run in tqdm(range(runs), desc=f"Runs for num_experts = {config.experts.num_experts}"):
        arranger = Arranger(config)
        expert_datasets, eval_data, test_data = arranger.create_datasets()
        logger.info(f"Len of Train data {len(expert_datasets[0])}")

        num_experts = config.experts.num_experts
        experts = []

        for i in range(num_experts):
            expert = Expert(config, i, expert_datasets[i], eval_data)
            logger.info(f"Ready expert {i}")
            experts.append(expert)  

        team = ExpertTeam(experts)
        metrics = Metrics()

        test_data = test_data.select(range(10))
        test_tasks = [task_set["dialogue"] for task_set in test_data]
        test_ground_truths = [task_set["summary"] for task_set in test_data]

        before_expert_scores = expert_test_evaluation(team, test_tasks, test_ground_truths, metrics)

        for i in range(num_experts):
            expert = experts[i]
            try:
                start_time = time()
                expert.fine_tune_std_lora(save=True)
                end_time = time()
                logger.info(f"Fine-tuned expert {i} in {end_time - start_time:.4f} seconds")
            except Exception as e:
                logger.error(f"Error fine-tuning expert {i}: {e}")
                raise e 

        team = ExpertTeam(experts)
        after_expert_scores = expert_test_evaluation(team, test_tasks, test_ground_truths, metrics)

        for i in range(len(before_expert_scores)):
            for expert_idx in range(len(before_expert_scores[i])):
                data = pd.concat([data, pd.DataFrame({
                    "run": [run+1],
                    "expert_id": [expert_idx],
                    "before": [before_expert_scores[i][expert_idx]],
                    "after": [after_expert_scores[i][expert_idx]]
                })], ignore_index=True)
    
    data.to_csv(hydra_output_path + "/expert_run_performance.csv", index=False)
    plot_expert_run_performance(data, hydra_output_path + f"/{config.experts.num_experts}_experts_run_performance.png")
    plot_expert_summary(data, hydra_output_path + f"/{config.experts.num_experts}_experts_summary.png")

if __name__ == "__main__":
    main()