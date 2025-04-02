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
        logger.info(f"Ground Truth: {ground_truth}")

        
        if self.config.data.category == "math":
            instruction = f"""As a teacher, guide the experts so that their answers get closer to the provided ground truth.
                The experts are given a math problem and their job is to solve it. Give a one lined instruction to each expert to improve their answers.
                The instructions should be in the following format:
                Expert : <instruction>
            """
        elif self.config.data.category == "translation":
            instruction = f"""List down the words in the expert answer that the expert was not able to translate correctly. The expert was given a task to translate a german sentence to english.
                Only list the words that expert could not translate correctly. The answer should be in the following format:
                <german word> : <english word>
                <german word> : <english word>
                ...
            """
        elif self.config.data.category == "summarization":
            instruction = f"""As a teacher, guide the experts so that their answers get closer to the provided ground truth.
                Give a one lined instruction to each expert to improve their answers. The instructions should be very generic, do not include specific details from the ground truth.
                The instructions should be in the following format:
                Expert : <instruction>
            """
        else:
            raise ValueError(f"Invalid category: {self.config.data.category}")

        
        prompt = f"{instruction}\n\n=== Expert Answers ===\n\n"
        for i, answer in enumerate(expert_answers):
            prompt += f"Expert {i}: {expert_answers[i]}\n\n"

        prompt += f"=== Task ===\n{task}\n\n"

        prompt += f"=== Ground Truth === \n {ground_truth}\n\n === Feedback ===\n"

        # logger.info(f"Prompt to Critic: {prompt}")
        tokenized_prompt = self.tokenizer(prompt, return_tensors="pt", padding=True)
        tokenized_prompt = tokenized_prompt.to(self.device_available)

        torch.cuda.empty_cache()
        output = self.model.generate(
            input_ids=tokenized_prompt["input_ids"],
            attention_mask=tokenized_prompt["attention_mask"],  
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=self.config.model_params.max_new_tokens
        )
        critic_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # logger.info(f"critic output whole: {critic_output}")
        critic_output = critic_output.split("=== Feedback ===")[-1].strip()
        logger.info(f"critic output whole: {critic_output}")
        
        return critic_output




        