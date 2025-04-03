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
           instruction = f""" Identify and list only the German words from the original sentence that the expert failed to translate correctly in their answer. Get the correct english translation from the provided Ground Truth.
                The answer should be in the following format: 
                <german word> : <english word>
                <german word> : <english word>
                ....
                Do not include the words that were translated correctly."""
        elif self.config.data.category == "summarization":
            instruction = f"""As a teacher, guide the expert so that its answer gets closer to the provided ground truth.
                Give a one lined instruction to the expert to improve its answer. The instructions should be very generic, do not include specific details from the ground truth.
                The instructions should be in the following format:
                Expert : <instruction>
            """
        else:
            raise ValueError(f"Invalid category: {config.data.category}")


        prompt = f"{instruction}\n\n"
        prompt += f"German Sentence: {task}\n\n"

        prompt += f"Expert Answers: \n"
        for i, answer in enumerate(expert_answers):
            prompt += f"Expert {i}: {expert_answers[i]}\n\n"

        prompt += f"Ground Truth: {ground_truth}\n\n"
        if self.config.data.category == "translation":
            prompt += f"=== Incorrect German words ===\n"
        elif self.config.data.category == "summarization":
            prompt += f"=== Feedback ===\n"
        elif self.config.data.category == "math":
            prompt += f"=== Feedback ===\n"
        else:
            raise ValueError(f"Invalid category: {self.config.data.category}")  
        # prompt += f"Provide a maximum of one line feedback for the experts here. \n"

        # logger.info(f"Prompt to Critic: {prompt}")
        tokenized_prompt = self.tokenizer(prompt, return_tensors="pt", padding=True)
        tokenized_prompt = tokenized_prompt.to(self.device_available)

        torch.cuda.empty_cache()
        output = self.model.generate(
            input_ids=tokenized_prompt["input_ids"],
            attention_mask=tokenized_prompt["attention_mask"],  
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=self.config.model_params.max_new_tokens,
            temperature=self.config.model_params.temperature,
            do_sample=self.config.model_params.do_sample,
            top_p=self.config.model_params.top_p,
            num_return_sequences=self.config.model_params.num_return_sequences,
            min_new_tokens=self.config.model_params.min_new_tokens
        )
        critic_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        if self.config.data.category == "translation":
            critic_output = critic_output.split("=== Incorrect German words ===")[-1].strip()
            logger.info(f"critic output whole: {critic_output}")

            result = {}
            lines = critic_output.strip().split("\n")
            for line in lines:
                try:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        result[key.strip()] = value.strip()
                    else:
                        pass
                except Exception as e:
                    logger.debug(f"Error processing line: '{line}' -> {e}")
                

        
            logger.info(f"Final critic output: {result}")
            return result

        else:
            critic_output = critic_output.split("=== Feedback ===")[-1].strip()
            return critic_output


        
        # expert_segments = critic_output.split("Expert")[1:]
        # matches = []
        
        # for segment in expert_segments:
        #     if match := re.match(r'\s*(\d+)\s*:\s*([^E]+)', segment):
        #         expert_num, feedback = match.groups()
        #         matches.append((expert_num, feedback.strip()))
        
        # logger.debug(f"Matches: {matches}")
        
        # if len(matches) == 0:
        #     return {num: "" for num in range(len(expert_answers))}

        # output_dict = {int(num): statement.strip() for num, statement in matches}
        
        # if len(output_dict) != len(expert_answers):
        #     logger.warning(f"Missing feedback for some experts. Expected {len(expert_answers)}, got {len(output_dict)}")
        #     for i in range(len(expert_answers)):
        #         if i not in output_dict:
        #             output_dict[i] = ""

        # logger.info(f"Critic output: {output_dict}")
        # logger.debug(f"Critic output completed")





        