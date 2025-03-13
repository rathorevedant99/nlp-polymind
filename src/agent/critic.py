"""
Author: Arushi Agrawal
Critic Class
"""

from src.agent.base import BaseAgent
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)

class Critic(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
    
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
        
        logger.info(f"Expert answers to critic: {expert_answers}")

        instruction = (
            "You are a critic. You have been given a list of answers by various experts, "
            "along with the ground truth for the given task. You have to evaluate them and "
            "return your answer in the following format: "
            "'Best Expert: <BEST_EXPERT>'"
            "Reasoning: <REASONING>"
        )

        prompt = f"{instruction}\n\nTask: {task}\n\nExpert Answers:\n\n"
        for i, answer in enumerate(expert_answers):
            prompt += f"Expert {i+1}: {expert_answers[i]}\n\n"

        prompt += f"Ground Truth: {ground_truth}\n\nAnswer:"

        logger.info(f"Prompt to critic: {prompt}")

        tokenized_prompt = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)

        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

        output = model.generate(
            input_ids=tokenized_prompt["input_ids"].to(model.device),
            attention_mask=tokenized_prompt["attention_mask"].to(model.device),  
            pad_token_id=tokenizer.eos_token_id 
        )
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info(f"Critic answer: {decoded_output}")
        return decoded_output
        # return self.tokenizer.decode(output[0], skip_special_tokens=True)




        