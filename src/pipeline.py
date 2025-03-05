"""
Author: Payal Agarwal
Pipeline Class
"""
from src.agents.Expert import ExpertTeam  
from src.agents.Critic import Critic  
import logging

def run_pipeline(config, task, ground_truth):
    logger = logging.getLogger("Pipeline")
    
    logger.info("Pipeline execution started.")
    expert_team = ExpertTeam(config)
    critic = Critic(config)
    
    expert_answers = expert_team.get_expert_answers(task)
    
    logger.info("Expert Answers:")
    for i, answer in enumerate(expert_answers):
        logger.info(f"Expert {i+1}: {answer}\n")

    evaluation = critic.evaluate(task, expert_answers, ground_truth)
    
    logger.info("Critic Evaluation: {evaluation}\n")

    logger.info("Pipeline execution started.")

    return expert_answers, evaluation
