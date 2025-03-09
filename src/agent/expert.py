"""
Author: Payal Agarwal
Expert Class
"""
from src.agent.base import BaseAgent
from peft import get_peft_model, LoraConfig, TaskType
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
import os
import logging
import hashlib

logger = logging.getLogger(__name__)

class Expert(BaseAgent):
    def __init__(self, config, expert_id, train_data=None, eval_data=None):
        super().__init__(config)
        self.expert_id = expert_id

        self.train_data = train_data
        self.eval_data = eval_data

        ## Adding this explicitly since Hydraconfig makes the target_modules non JSON serializable
        target_modules = list(config.lora.target_modules) if config.lora.target_modules else None 

        self.lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            lora_dropout=config.lora.lora_dropout,
            target_modules=target_modules
        )

        if config.lora.enabled:
            self.model = get_peft_model(self.model, self.lora_config)

        self.model.config.gradient_checkpointing = False
        self.model.print_trainable_parameters()

        self.training_args = TrainingArguments(**config.training)

        self.feedback = []
    
    def fine_tune_unsloth(self):
        """
        Fine-tune the critic on the given data using Unsloth.
        """
        raise NotImplementedError("Unsloth fine-tuning is not implemented yet")

    def fine_tune_std_lora(self, save=False):
        """
        Fine-tune the critic on the given data using standard LORA.
        """
        if self.train_data is None or self.eval_data is None:
            raise ValueError("Train and eval data must be provided")
        
        if os.path.exists(self.config.training.output_dir+f"/expert_{self.expert_id}"):
            self.load_lora()
            logger.info(f"Loaded LORA weights for expert {self.expert_id}")
            return
        
        logger.info(f"Fine-tuning expert {self.expert_id} using standard LORA")
        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_data,
            eval_dataset=self.eval_data,
            data_collator=data_collator
        )

        self.trainer.train()
        logger.info(f"Trained model")
        if save:
            self.store_lora()
            logger.info(f"Stored LORA weights")

    def store_lora(self):
        """
        Store the LORA weights.
        """
        if not os.path.exists(self.config.training.output_dir+f"/expert_{self.expert_id}"):
            os.makedirs(self.config.training.output_dir+f"/expert_{self.expert_id}")
        self.model.save_pretrained(self.config.training.output_dir+f"/expert_{self.expert_id}")

    def load_lora(self):
        """
        Load the LORA weights.
        """
        try:
            self.model.load_adapter(model_id=self.config.training.output_dir+f"/expert_{self.expert_id}", adapter_name="default")
        except Exception as e:
            logger.error(f"Error loading LORA weights for expert {self.expert_id}: {e}", exc_info=True)
            raise e

    def generate(self, task):
        """
        Generate an expert answer for the given task.
        Args:
            task (str): Task description
        Returns:
            str: Generated expert answer
        """
        expert_prompt = f"You are an expert. Provide a detailed and accurate answer to the following task:\n\nTask: {task}\n\nAnswer:"
        
        tokenized_prompt = self.tokenizer(expert_prompt.format(task), return_tensors="pt", truncation=True, padding=True)
        output = self.model.generate(
            input_ids=tokenized_prompt["input_ids"].to(self.model.device),
            attention_mask=tokenized_prompt["attention_mask"].to(self.model.device),  
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=128
        )
        
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
    
    def update(self, feedback):
        """
        Update the expert's feedback.
        """
        self.feedback.append(feedback)