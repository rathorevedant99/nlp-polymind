from src.agent.team import ExpertTeam
from src.agent.critic import Critic
from src.metrics import Metrics
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class Debate:
    def __init__(self, config, expert_team: ExpertTeam, critic: Critic):
        self.config = config
        self.debate_rounds = config.experts.debate_rounds
        self.expert_team = expert_team
        self.critic = critic
        self.metric_dict = {}
        self._have_debated = False
        self.batch_size = config.experts.batch_size
        
        if not isinstance(expert_team, ExpertTeam):
            raise ValueError("Expert team must be an instance of ExpertTeam class")
        
        if not isinstance(critic, Critic):
            raise ValueError("Critic must be an instance of Critic class")

    def execute_debate(self, tasks: List[str], ground_truths: List[str]):
        """
        Execute a debate between the experts and the critic.
        """
        metrics = Metrics()

        for debate_round in range(self.debate_rounds):
            logger.info(f"Debate round {debate_round+1} started")

            task_batch = tasks[debate_round:debate_round+self.batch_size]
            ground_truth_batch = ground_truths[debate_round:debate_round+self.batch_size]

            expert_answers = self.expert_team.get_expert_answers(task_batch)
            critic_answer = self.critic(task_batch, expert_answers, ground_truth_batch)

            for expert in self.expert_team.experts:
                expert.update(critic_answer)

            rouge_scores, bertscore_scores, novelty_scores, length_ratios = metrics(ground_truths[debate_round], expert_answers)
            self.metric_dict[f"{debate_round+1}"] = {
                "rouge_scores": rouge_scores,
                "bertscore_scores": bertscore_scores,
                "novelty_scores": novelty_scores,
                "length_ratios": length_ratios
            }
            logger.info(f"Debate round {debate_round+1} completed")

        
        self._have_debated = True
        logger.info(f"Debate completed")
        logger.debug(f"Metric dict: {self.metric_dict}")
    
    def get_final_answer(self, task: str):
        """
        Get the final answer after the debate.
        """
        if not self._have_debated:
            raise ValueError("Debate has not been executed yet")
        self.expert_answers = self.expert_team.get_expert_answers(task)
        return self.expert_answers
    
    def get_first_answer(self):
        """
        Get the first answer from the experts.
        """
        return self.first_answer