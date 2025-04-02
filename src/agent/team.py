"""
Author: Vedant S Rathore
Team Class
"""

from src.agent.expert import Expert
from typing import List
class ExpertTeam:
    def __init__(self, experts: List[Expert]):
        self.experts = experts

        if not all(isinstance(expert, Expert) for expert in self.experts):
            raise ValueError("All experts must be instances of Expert class")
    
    def __call__(self, task):
        return self.get_expert_answers(task)
    
    def add_expert(self, expert):
        """
        Add an expert to the team.
        """
        if not isinstance(expert, Expert):
            raise ValueError("Expert must be an instance of Expert class")
        self.experts.append(expert)
    
    def get_expert_answers(self, task, feedback=False):
        """
        Get answers from all experts for a given task.
        Args:
            task (str): Task description
        Returns:
            model_answers (dict): Dictionary containing answers from all experts
        """
        model_answers = {}
        for expert in self.experts:
            model_answers[expert.expert_id] = expert.generate(task, feedback)
        return model_answers