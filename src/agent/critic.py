"""
Author: Arushi Agrawal
Critic Class
"""

from agent.base import BaseAgent

class Critic(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
    

    def evaluate(self, task, expert_answers, ground_truth):
        """
        Evaluate expert answers and return the best one along with reasoning
        Args:
            task (str): Task description
            expert_answers (List[str]): List of expert answers
            ground_truth (str): Ground truth answer
        Returns:
            str: Best answer along with reasoning
        """
        
        instruction = (
            "You are a critic. You have been given a list of answers by various experts, "
            "along with the ground truth for the given task. You have to evaluate them and "
            "return the one that is closest to the ground truth. Provide a reasoning for your choice, "
            "and also provide insights on the other answers. Keep in mind that the goal is to provide "
            "constructive feedback to the experts. Keep it short and concise."
        )

        prompt = f"{instruction}\n\nTask: {task}\n\nExpert Answers:\n\n"
        for i, answer in enumerate(expert_answers):
            prompt += f"Expert {i+1}: {answer}\n\n"

        prompt += f"Ground Truth: {ground_truth}\n\nAnswer:"

        tokenized_prompt = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)

        output = self.model.generate(
            input_ids=tokenized_prompt["input_ids"],
            attention_mask=tokenized_prompt["attention_mask"],  
            pad_token_id=self.tokenizer.eos_token_id 
        )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)




        