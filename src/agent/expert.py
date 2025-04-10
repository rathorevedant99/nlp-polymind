"""
Author: Payal Agarwal
Expert Class
"""
from src.agent.base import BaseAgent
from peft import get_peft_model, LoraConfig, TaskType
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset
import os
import logging
import torch
import random
from typing import List
logger = logging.getLogger(__name__)

class Expert(BaseAgent):
    def __init__(self, config, expert_id, train_data=None, eval_data=None):
        super().__init__(config, "expert")
        self.expert_id = expert_id
        if self.config.data.category == "summarization":
            self.default_prompt = "Summarize this conversation:\n\n{}\n\n"
        elif self.config.data.category == "math":
            self.default_prompt = "Solve this math problem:\n\n{}\n\n"
        elif self.config.data.category == "translation":
            self.default_prompt = "Translate this conversation from German to English:\n\n{}\n\n"
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

        ## The random seed prevents the finetuning phase from fixing the global seed
        self.training_args = TrainingArguments(**config.training, seed=random.randint(0, 2**32 - 1))
        self.feedback_size = config.experts.feedback_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def fine_tune_unsloth(self):
        """
        Fine-tune the critic on the given data using Unsloth.
        """
        raise NotImplementedError("Unsloth fine-tuning is not implemented yet")

    def fine_tune_std_lora(self, save=False, load=False):
        """
        Fine-tune the critic on the given data using standard LORA.
        """
        if self.train_data is None or self.eval_data is None:
            raise ValueError("Train and eval data must be provided")
        
        if os.path.exists(self.config.training.output_dir+f"/expert_{self.expert_id}"):
            if load:
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

    def store_lora(self, add_to_name=""):
        """
        Store the LORA weights.
        """
        if not os.path.exists(self.config.training.output_dir+f"/expert_{self.expert_id}{add_to_name}"):
            os.makedirs(self.config.training.output_dir+f"/expert_{self.expert_id}{add_to_name}")
        self.model.save_pretrained(self.config.training.output_dir+f"/expert_{self.expert_id}{add_to_name}")

    def load_lora(self, add_to_name=""):
        """
        Load the LORA weights.
        """
        try:
            self.model.load_adapter(model_id=self.config.training.output_dir+f"/expert_{self.expert_id}{add_to_name}", adapter_name="default")
        except Exception as e:
            logger.error(f"Error loading LORA weights for expert {self.expert_id}: {e}", exc_info=True)
            raise e

    def generate(self, task: str):
        """
        Generate an expert answer for the given task.
        Args:
            task (str): Task description
        Returns:
            str: Generated expert answer
        """
        expert_prompt = self.default_prompt.format(task)
        logger.debug(f"Expert {self.expert_id} prompt: {expert_prompt}")
        
        tokenized_prompt = self.tokenizer(expert_prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        self.model.eval()
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids=tokenized_prompt["input_ids"].to(self.model.device),
                attention_mask=tokenized_prompt["attention_mask"].to(self.model.device),
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.config.model_params.max_new_tokens,
                temperature=self.config.model_params.temperature,
                do_sample=self.config.model_params.do_sample,
                top_p=self.config.model_params.top_p,
                num_return_sequences=self.config.model_params.num_return_sequences,
                min_new_tokens=self.config.model_params.min_new_tokens
            )
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        logger.debug(f"Expert {self.expert_id} answer: {decoded_output}")
        
        return decoded_output
    
    def memory_fine_tuning(self, instruction_data):
        """
        Continue fine-tuning the expert on the given instruction data from memory.
        """
        def preprocess_function(examples):
            # Process each example in the batch
            inputs = [instruction + "\n" + inp for instruction, inp in zip(examples['instruction'], examples['input'])]
            targets = examples['output']
            
            model_inputs = self.tokenizer(
                inputs, 
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            labels = self.tokenizer(
                targets,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        dataset = Dataset.from_dict({
            'instruction': [x['instruction'] for x in instruction_data],
            'input': [x['input'] for x in instruction_data],
            'output': [x['output'] for x in instruction_data]
        })

        dataset = dataset.map(preprocess_function, batched=True)

        future_training_args = TrainingArguments(
            save_strategy="no",
            learning_rate=self.config.training.learning_rate * 0.1,  # Lower learning rate for continued training
            max_steps=200,
            logging_steps=50,
            seed=random.randint(0, 2**32 - 1)
        )

        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

        self.trainer = Trainer(
            model=self.model,
            args=future_training_args,
            train_dataset=dataset,
            eval_dataset=self.eval_data,
            data_collator=data_collator
        )

        self.trainer.train()
        # self.store_lora(add_to_name="_continued_ft")

    def update(self, feedback):
        """
        Update the expert's feedback.
        """
        self.feedback.append(feedback)