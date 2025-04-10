"""
Author: Vedant S Rathore
Base Class for all LLM Agents
"""
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from src.openrouter import OpenRouterClient
import torch

class BaseAgent:
    def __init__(self, config, agent_type="expert"):
        self.config = config
        if agent_type=="expert":
            if self.config.experts.type == "causal":
                self.model = AutoModelForCausalLM.from_pretrained(self.config.experts.name)
            elif self.config.experts.type == "seq2seq":
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.experts.name)
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.experts.name)
        
        elif agent_type=="critic":
            if self.config.critic.is_openrouter:
                self.client = OpenRouterClient()
            else:
                if self.config.critic.type == "causal":
                    self.model = AutoModelForCausalLM.from_pretrained(self.config.critic.name)
                elif self.config.critic.type == "seq2seq":
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.critic.name)
                self.device_available = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
                self.model.to(self.device_available)
                self.model.eval()

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.critic.name)
        
        else:
            raise ValueError(f"Invalid agent type: {self.config.agent.type}")

        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt):
        return self.model.generate(prompt)

    def __call__(self, prompt):
        return self.generate(prompt)