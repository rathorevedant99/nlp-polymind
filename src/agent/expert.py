"""
Author: Payal Agarwal
Expert Class
"""
from agent.base import BaseAgent

class Expert(BaseAgent):
    def __init__(self, config, expert_id):
        super().__init__(config)
        self.expert_id = expert_id

    def generate(self, prompt):
        """
        Generate an expert answer for the given task.
        Args:
            task (str): Task description
        Returns:
            str: Generated expert answer
        """
        expert_prompt = f"You are an expert. Provide a detailed and accurate answer to the following task:\n\nTask: {task}\n\nAnswer:"
        
        tokenized_prompt = self.tokenizer(expert_prompt, return_tensors="pt", truncation=True, padding=True)
        output = self.model.generate(
            input_ids=tokenized_prompt["input_ids"],
            attention_mask=tokenized_prompt["attention_mask"],  
            pad_token_id=self.tokenizer.eos_token_id 
        )
        
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

class ExpertTeam:
    def __init__(self, config):
        self.expert1 = Expert(config, expert_id=1)
        self.expert2 = Expert(config, expert_id=2)
    
    def get_expert_answers(self, task):
        """
        Get answers from both experts for a given task.
        Args:
            task (str): Task description
        Returns:
            List[str]: List containing answers from both experts
        """
        answer1 = self.expert1.generate(task)
        answer2 = self.expert2.generate(task)
        return [answer1, answer2]