"""
Author: Arushi Agrawal
Critic Class
"""

from src.agent.base import BaseAgent

class Critic(BaseAgent):
    def __init__(self, config):
        super().__init__(config, "critic")
        self.task_type = config.data.category
    
    def __call__(self, task, expert_answers, ground_truth):
        """
        Evaluate expert answers and return the best one along with reasoning
        """
        return self.evaluate(task, expert_answers, ground_truth)
    
    def __repr__(self):
        """
        Return a string representation of the critic
        """
        return f"Critic(model={self.model}, tokenizer={self.tokenizer})"

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
        
        # instruction = (
        #     "You are a critic. You have been given a list of answers by various experts, "
        #     "along with the ground truth for the given task. You have to evaluate them and "
        #     "return the one that is closest to the ground truth. Provide a reasoning for your choice, "
        #     "and also provide insights on the other answers. Keep in mind that the goal is to provide "
        #     "constructive feedback to the experts. Keep it short and concise."
        # )

        instruction = f"""For this {self.task_type} task, compare the expert answers to the ground truth.
            What could be improved in each answer?
            Focus on {self.task_type}-specific feedback.
            Be brief and specific."""

        prompt = f"{instruction}\n\n=== Task === {task}\n\n=== Expert Answers ===\n\n"
        for i, answer in enumerate(expert_answers):
            prompt += f"Expert {i}: {answer}\n\n"

        prompt += f"=== Ground Truth === \n {ground_truth}\n\n === Feedback ===\n"

        tokenized_prompt = self.tokenizer(prompt, return_tensors="pt", padding=True)

        output = self.model.generate(
            input_ids=tokenized_prompt["input_ids"].to(self.model.device),
            attention_mask=tokenized_prompt["attention_mask"].to(self.model.device),  
            pad_token_id=self.tokenizer.eos_token_id,
            max_length=512
        )
        critic_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Critic output: {critic_output}")

        return critic_output




        