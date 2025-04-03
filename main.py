"""
Starting project
"""
import hydra
from omegaconf import DictConfig
from src.agent.expert import Expert
from src.agent.team import ExpertTeam
from src.agent.critic import Critic
from src.utils.arranger import Arranger
from src.utils.plotmetrics import Plotter
from src.eval import Debate
from src.metrics import Metrics
from src.memory import Memory
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from huggingface_hub import login


@hydra.main(version_base=None, config_path="configs", config_name="config.yaml")
def main(config: DictConfig):
    """
    Loads the config and runs the experiment.
    """
    hydra_output_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
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

    shuffled_eval_data = eval_data.shuffle()
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

    test_data = test_data.select(range(5))
    test_tasks = [task_set["dialogue"] for task_set in test_data]
    test_ground_truths = [task_set["summary"] for task_set in test_data]

    expert_answers = {}
    before_expert_scores = {}
    for i, task in enumerate(test_tasks):
        expert_answers[i] = team.get_expert_answers(task)
        rouge_score = metrics.eval_rouge(test_ground_truths[i], expert_answers[i])
        expert0_rouge = rouge_score["0"]["rouge1"].fmeasure
        expert1_rouge = rouge_score["1"]["rouge1"].fmeasure
        expert2_rouge = rouge_score["2"]["rouge1"].fmeasure
        before_expert_scores[i] = [expert0_rouge, expert1_rouge, expert2_rouge]

    expert0_mean_rouge1_before = sum(before_expert_scores[i][0] for i in range(len(before_expert_scores))) / len(before_expert_scores)
    expert1_mean_rouge1_before = sum(before_expert_scores[i][1] for i in range(len(before_expert_scores))) / len(before_expert_scores)
    expert2_mean_rouge1_before = sum(before_expert_scores[i][2] for i in range(len(before_expert_scores))) / len(before_expert_scores)
        
    memory = debate.execute_debate(tasks, ground_truths)
    del debate
    del critic
    # memory = Memory()
    # memory.load_feedback()

    instruction_data = memory.provide_instruction_data()

    for expert in experts:
        expert.memory_fine_tuning(instruction_data)

    expert_answers = {}
    after_expert_scores = {}
    for i, task in enumerate(test_tasks):
        expert_answers[i] = team.get_expert_answers(task)
        rouge_score = metrics.eval_rouge(test_ground_truths[i], expert_answers[i])
        expert0_rouge = rouge_score["0"]["rouge1"].fmeasure
        expert1_rouge = rouge_score["1"]["rouge1"].fmeasure
        expert2_rouge = rouge_score["2"]["rouge1"].fmeasure
        after_expert_scores[i] = [expert0_rouge, expert1_rouge, expert2_rouge]
        
    expert0_mean_rouge1_after = sum(after_expert_scores[i][0] for i in range(len(after_expert_scores))) / len(after_expert_scores)
    expert1_mean_rouge1_after = sum(after_expert_scores[i][1] for i in range(len(after_expert_scores))) / len(after_expert_scores)
    expert2_mean_rouge1_after = sum(after_expert_scores[i][2] for i in range(len(after_expert_scores))) / len(after_expert_scores)

    print(f"Expert 0 mean ROUGE-1 before: {expert0_mean_rouge1_before}")
    print(f"Expert 1 mean ROUGE-1 before: {expert1_mean_rouge1_before}")
    print(f"Expert 2 mean ROUGE-1 before: {expert2_mean_rouge1_before}")

    print(f"Expert 0 mean ROUGE-1 after: {expert0_mean_rouge1_after}")
    print(f"Expert 1 mean ROUGE-1 after: {expert1_mean_rouge1_after}")
    print(f"Expert 2 mean ROUGE-1 after: {expert2_mean_rouge1_after}")


    # plotter = Plotter(config, debate.metric_dict)
    # plotter(hydra_output_path)


if __name__ == "__main__":
    main()