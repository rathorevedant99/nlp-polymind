"""
Author: Vedant S Rathore
Base Class for all LLM Agents
"""
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

class BaseAgent:
    def __init__(self, config):
        self.config = config
        if self.config.agent.type == "causal":
            self.model = AutoModelForCausalLM.from_pretrained(self.config.agent.name)
        elif self.config.agent.type == "seq2seq":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.agent.name)
        else:
            raise ValueError(f"Invalid agent type: {self.config.agent.type}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.agent.name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt):
        return self.model.generate(prompt)

    def __call__(self, prompt):
        return self.generate(prompt)