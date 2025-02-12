import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import evaluate

"""This is where you get the model you want. I have used the t5-small model here.
You can use any other model you want. Just make sure to use the correct tokenizer
and model class."""
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

"""
PEFT stands for Parameter Efficient Fine-Tuning. It is a technique that allows you to
fine-tune a model with fewer parameters while maintaining the same performance. This is
achieved by removing certain parameters from the model and replacing them with a
parameter-efficient module. The PEFT module is a small neural network that is used to
replace the removed parameters. The PEFT module is trained to predict the removed
parameters based on the input data. This allows the model to learn to perform the task
with fewer parameters.

LoRA here is a PEFT module that is used to replace the self-attention mechanism in the
transformer model. The target modules are the query and value matrices in the
self-attention mechanism. r is the rank of the lora matrix. """
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=4, 
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q", "v"],
)

"""The get_peft_model function is used to get the PEFT model. It takes the base model
and the PEFT configuration as input and returns the PEFT model. The PEFT model is the
base model with the PEFT module inserted in place of the target modules."""
model = get_peft_model(model, peft_config)
model.config.gradient_checkpointing = True # Enable gradient checkpointing for memory efficiency during training. This slows down the evaluation process.
model.print_trainable_parameters()


dataset = load_dataset("samsum")
# dataset = load_dataset("samsum", split="train[:10%]+validation[:5%]+test[:5%]") # Use a smaller dataset for prototyping

# Remove entries with empty dialogue or summary
dataset = dataset.filter(lambda x: len(x['dialogue']) > 0 and len(x['summary']) > 0)

def preprocess_data(examples, tokenizer, max_input_length=512, max_target_length=150):
    inputs = [ex for ex in examples['dialogue']]
    targets = [ex for ex in examples['summary']]
    
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding="max_length"
    )
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            truncation=True,
            padding="max_length"
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize the dataset
tokenized_datasets = dataset.map(preprocess_data, fn_kwargs={"tokenizer": tokenizer}, batched=True)

train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

#TODO: Implement compute_metrics
metric = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {k: v.mid.fmeasure * 100 for k, v in result.items()}  # Convert to percentage

# Training arguments
# training_args = TrainingArguments(
#     output_dir="./results",
#     evaluation_strategy="steps",
#     eval_steps=100,
#     per_device_train_batch_size=1,
#     per_device_eval_batch_size=1,
#     logging_dir="./logs",
#     logging_steps=100,
#     save_steps=100,
#     save_total_limit=2,
#     num_train_epochs=1,
#     report_to="none",
    # gradient_accumulation_steps=4,
    # eval_accumulation_steps=1,
# )

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy= "steps",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    eval_accumulation_steps=1,
    eval_steps=50,
    logging_steps=50,
    max_steps=500,
    # gradient_checkpointing=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # data_collator=data_collator,
    # compute_metrics=compute_metrics,
)

# print("Evaluating base model...")
# eval_results = trainer.evaluate()
# print("Base model ROUGE scores:", eval_results)

# Train model
trainer.train()

# Evaluate final model after training
# print("Evaluating final model...")
# eval_results_final = trainer.evaluate()
# print("Final model ROUGE scores:", eval_results_final)