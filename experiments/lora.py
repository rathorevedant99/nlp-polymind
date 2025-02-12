import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import evaluate

# Load a small T5 model
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Apply LoRA to the model
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,  # Seq2Seq task
    r=4,  # Rank of LoRA update matrices
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1,  # Dropout for LoRA layers
    target_modules=["q", "v"],  # Apply LoRA to attention key layers
)
model = get_peft_model(model, peft_config)
model.config.gradient_checkpointing = True
model.print_trainable_parameters()

# Load the samsum dataset
dataset = load_dataset("samsum")
# dataset = load_dataset("samsum", split="train[:10%]+validation[:5%]+test[:5%]") # Use a smaller dataset for faster training

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

# Load ROUGE metric
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