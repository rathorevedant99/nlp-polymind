"""
Author: Vedant S Rathore
Base Class for all LLM Agents
"""
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from peft import get_peft_model, LoraConfig, TaskType
class BaseAgent:
    def __init__(self, config):
        self.config = config
        if self.config.agent.type == "causal":
            self.model = AutoModelForCausalLM.from_pretrained(self.config.agent.name)
        elif self.config.agent.type == "seq2seq":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.agent.name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.agent.name)
        if self.config.lora.enabled:
            self.model = get_peft_model(self.model, self.config.lora)
            self.model.print_trainable_parameters()

    def generate(self, prompt):
        return self.model.generate(prompt)

    def __call__(self, prompt):
        return self.generate(prompt)