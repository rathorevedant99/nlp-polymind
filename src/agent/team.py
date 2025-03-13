"""
Author: Vedant S Rathore
Team Class
"""

from src.agent.expert import Expert

class ExpertTeam:
    def __init__(self, experts=None):
        self.experts = experts if experts is not None else []

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
    
    def get_expert_answers(self, task):
        """
        Get answers from all experts for a given task.
        Args:
            task (str): Task description
        Returns:
            model_answers (dict): Dictionary containing answers from all experts
        """
        model_answers = {}
        for expert in self.experts:
            model_answers[expert.expert_id] = expert.generate(task)
        return model_answers