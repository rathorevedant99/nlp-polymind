from src.agent.team import ExpertTeam
from src.agent.critic import Critic
from src.metrics import Metrics
from typing import List, Dict
import logging
from src.memory import Memory
from tqdm import tqdm

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
        memory = Memory()

        batched_tasks = [tasks[i:i+self.batch_size] for i in range(0, len(tasks), self.batch_size)]
        batched_ground_truths = [ground_truths[i:i+self.batch_size] for i in range(0, len(ground_truths), self.batch_size)]

        counter = 0
        max_rounds = self.config.experts.debate_rounds

        for task_batch, ground_truth_batch in tqdm(zip(batched_tasks, batched_ground_truths), total=max_rounds, desc="Debate"):
            expert_answers = {}
            for task in task_batch:
                expert_answers[task] = self.expert_team.get_expert_answers(task)

            expert_answers = [expert_answers[task] for task in task_batch]
            
            critic_feedback = self.critic(task_batch, expert_answers, ground_truth_batch)

            memory.add_critic_feedback(original_inputs=task_batch, expert_outputs=expert_answers, critic_feedback=critic_feedback)
            
            counter += 1
            if counter == max_rounds:
                break

        return memory
    
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