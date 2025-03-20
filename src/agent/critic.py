"""
Author: Arushi Agrawal
Critic Class
"""

from src.agent.base import BaseAgent
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import torch

logger = logging.getLogger(__name__)

class Critic(BaseAgent):
    def __init__(self, config):
        super().__init__(config, "critic")
        self.task_type = config.data.category
        self.device_available = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(self.device_available)
    
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

        # logger.info(f"Model:{self.model}")

        instruction = f"""As a feedback provider, compare the provided expert answers with the provided ground truth.
            Give a one lined feedback on what could be improved in each expert answer to make it closer to the ground truth.
        """

        prompt = f"{instruction}\n\n=== Expert Answers ===\n\n"
        for i, answer in enumerate(expert_answers):
            prompt += f"Expert {i}: {expert_answers[i]}\n\n"

        prompt += f"=== Ground Truth === \n {ground_truth}\n\n === Provide Feedback ===\n"
        # prompt += f"Provide a maximum of one line feedback for the experts here. \n"

        logger.debug(f"Prompt to Critic: {prompt}")
        tokenized_prompt = self.tokenizer(prompt, return_tensors="pt", padding=True)
        tokenized_prompt = tokenized_prompt.to(self.device_available)

        output = self.model.generate(
            input_ids=tokenized_prompt["input_ids"],
            attention_mask=tokenized_prompt["attention_mask"],  
            pad_token_id=self.tokenizer.eos_token_id, max_length=512
        )
        critic_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        critic_output = critic_output.split("=== Provide Feedback ===")[-1].strip()
        logger.debug(f"critic output whole: {critic_output}")
        
        expert_segments = critic_output.split("Expert")[1:]
        matches = []
        
        for segment in expert_segments:
            if match := re.match(r'\s*(\d+)\s*:\s*([^E]+)', segment):
                expert_num, feedback = match.groups()
                matches.append((expert_num, feedback.strip()))
        
        logger.debug(f"Matches: {matches}")
        
        if len(matches) == 0:
            return {num: "No matches found in critic output" for num in range(len(expert_answers))}

        output_dict = {int(num): statement.strip() for num, statement in matches}
        
        if len(output_dict) != len(expert_answers):
            logger.warning(f"Missing feedback for some experts. Expected {len(expert_answers)}, got {len(output_dict)}")
            for i in range(len(expert_answers)):
                if i not in output_dict:
                    output_dict[i] = "No feedback provided"

        logger.debug(f"Critic output: {output_dict}")
        logger.debug(f"Critic output completed")
        return output_dict




        