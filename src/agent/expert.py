
"""
Author: Payal Agarwal
Expert Class
"""
from src.agent.base import BaseAgent
from peft import get_peft_model, LoraConfig, TaskType
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
import os
import logging
import torch

logger = logging.getLogger(__name__)

class Expert(BaseAgent):
    def __init__(self, config, expert_id, train_data=None, eval_data=None):
        super().__init__(config, "expert")
        self.expert_id = expert_id
        if self.config.data.category == "summarization":
            self.default_prompt = "Summarize this conversation:\n\n{}\n\n"
        elif self.config.data.category == "math":
            self.default_prompt = "Solve this math problem:\n\n{}\n\n"
        else:
            raise ValueError(f"Unsupported data category: {self.config.data.category}")

        self.train_data = train_data
        self.eval_data = eval_data

        if self.config.experts.type == "causal":
            target_modules = ["q_proj", "v_proj"]
        else:
            target_modules = ["q", "v"] # ["q" "v", "k", "o"]

        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM if self.config.experts.type == "causal" else TaskType.SEQ_2_SEQ_LM,
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            lora_dropout=config.lora.lora_dropout,
            target_modules=target_modules,
            bias=config.lora.bias
        )

        if config.lora.enabled:
            self.model = get_peft_model(self.model, self.lora_config)

        self.model.config.gradient_checkpointing = False
        self.model.print_trainable_parameters()

        self.training_args = TrainingArguments(**config.training)

        self.feedback = []
        self.feedback_size = config.experts.feedback_size
    
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
        feedback_context = ""
        if len(self.feedback) > 0:
            if len(self.feedback) > self.feedback_size:
                feedback_context += "\n".join([f"- {feedback}" for feedback in self.feedback[-self.feedback_size:]])
            else:
                feedback_context += "\n".join([f"- {feedback}" for feedback in self.feedback])
            feedback_context += "\n\n Consider the above information while generating the response.\n\n"
    
        expert_prompt = self.default_prompt.format(task) + feedback_context
        logger.info(f"Expert {self.expert_id} prompt: {expert_prompt}")
        
        tokenized_prompt = self.tokenizer(expert_prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        self.model.eval()
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids=tokenized_prompt["input_ids"].to(self.model.device),
                attention_mask=tokenized_prompt["attention_mask"].to(self.model.device),
                pad_token_id=self.tokenizer.eos_token_id, max_length=2000
            )

        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info(f"Expert {self.expert_id} answer: {decoded_output}")
        
        return decoded_output
    
    def update(self, feedback):
        """
        Update the expert's feedback.
        """
        relevant_feedback = {k: v for k, v in feedback.items() if k == self.expert_id}
        relevant_feedback = relevant_feedback[self.expert_id]
        logger.info(f"Expert {self.expert_id} feedback: {relevant_feedback}")
        self.feedback.append(relevant_feedback)