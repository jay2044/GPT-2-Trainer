from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch
import os

# loading GPT-2 model and tokenizer
model_name = "gpt2-xl"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# your dataset
data_path = "data.txt"  # Path to your data file
with open(data_path, "r", encoding="utf-8") as file:
    slang_texts = [{"text": line.strip()} for line in file if line.strip()]

# Convert to Hugging Face Dataset format
dataset = Dataset.from_list(slang_texts)

#Split the dataset into training and validation sets
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# preprocess the dataset
def preprocess_function(examples):
    # Tokenize each example
    inputs = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,  #Adjust
    )
    # set labels to be the same as input_ids for modeling language
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs

# Apply tokenization to the datasets
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_eval = eval_dataset.map(preprocess_function, batched=True)

# Set the format of the datasets for PyTorch
tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_eval.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Define training arguments
training_args = TrainingArguments(
    output_dir='./gpt2-slang',          # Output directory
    per_device_train_batch_size=4,      # Adjust based on GPU memory
    num_train_epochs=3,                  # Total number of training epochs
    learning_rate=5e-5,                  # Learning rate
    weight_decay=0.01,                   # Weight decay
    logging_dir='./logs',                # Directory for storing logs
    logging_steps=500,                   # Log every 500 steps
    eval_strategy="epoch",               # Evaluation strategy
    save_strategy="epoch",               # Save strategy
    load_best_model_at_end=True,         # Load the best model at the end
    metric_for_best_model="eval_loss",    # Metric to determine the best model
    greater_is_better=False,             # Lower loss is better
)

# Initialize
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
)

trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained('./gpt2') # Set desired name
tokenizer.save_pretrained('./gpt2')
