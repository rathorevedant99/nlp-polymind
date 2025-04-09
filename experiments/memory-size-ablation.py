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
import shutil
import json
import os

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
    data_json_path = hydra_output_path + "/run_data.json"

    for run in tqdm(range(runs), desc="Run"):
        if os.path.exists(data_json_path):
            with open(data_json_path, "r") as f:
                data_json = json.load(f)
        else:
            data_json = {}

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

        shuffled_eval_data = eval_data.shuffle(seed=run)
        if config.data.name == "samsum":
            tasks = [task_set["dialogue"] for task_set in shuffled_eval_data]
            ground_truths = [task_set["summary"] for task_set in shuffled_eval_data]
        elif config.data.name == "gsm8k":
            tasks = [task_set["question"] for task_set in shuffled_eval_data]
            ground_truths = [task_set["answer"] for task_set in shuffled_eval_data]
        elif config.data.name == "opus":
            tasks = [task_set["de"] for task_set in shuffled_eval_data]
            ground_truths = [task_set["en"] for task_set in shuffled_eval_data]
        else:
            raise ValueError(f"Invalid dataset name: {config.dataset_name}")
        
        metrics = Metrics()

        test_data = test_data.select(range(10))
        test_tasks = [task_set["dialogue"] for task_set in test_data]
        test_ground_truths = [task_set["summary"] for task_set in test_data]

        before_expert_scores = expert_test_evaluation(team, test_tasks, test_ground_truths, metrics)
            
        memory = debate.execute_debate(tasks, ground_truths, append=True)
        del debate
        del critic

        instruction_data = memory.provide_instruction_data()

        for expert in experts:
            expert.memory_fine_tuning(instruction_data)

        after_expert_scores = expert_test_evaluation(team, test_tasks, test_ground_truths, metrics)

        data_json[run+1] = {
            "before": before_expert_scores,
            "after": after_expert_scores
        }

        json.dump(data_json, open(data_json_path, "w"), indent=2)

        for i in range(len(before_expert_scores)):
            for expert_idx in range(len(before_expert_scores[i])):
                data = pd.concat([data, pd.DataFrame({
                    "run": [run+1],
                    "expert_id": [expert_idx],
                    "before": [before_expert_scores[i][expert_idx]],
                    "after": [after_expert_scores[i][expert_idx]]
                })], ignore_index=True)

    data.to_csv(hydra_output_path + "/expert_run_performance.csv", index=False)
    plot_expert_run_performance(data, hydra_output_path + "/succesive_memory_experts_run_performance.png")
    plot_expert_summary(data, hydra_output_path + "/succesive_memory_experts_summary.png")
    
    memory_path = "./memory" + f"/{config.data.category}/feedback_history.json"
    shutil.copy(memory_path, hydra_output_path + "/memory.json")

if __name__ == "__main__":
    main()