"""
Author: Vedant S Rathore
Team Class
"""

from src.agent.expert import Expert

class ExpertTeam:
    def __init__(self, experts=None):
        self.experts = experts if experts is not None else []
    
    def add_expert(self, expert):
        """
        Add an expert to the team.
        """
        if not isinstance(expert, Expert):
            raise ValueError("Expert must be an instance of Expert class")
        self.experts.append(expert)
    
    def get_expert_answers(self, task):
        """
        Get answers from both experts for a given task.
        Args:
            task (str): Task description
        Returns:
            List[str]: List containing answers from both experts
        """
        answers = [expert.generate(task) for expert in self.experts]
        return answers