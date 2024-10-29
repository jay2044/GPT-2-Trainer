# GPT-2 Trainer

## Overview
To use GPT-2 Trainer, start by training the model on your custom dataset. Simply add your data to `data.txt` and run `trainer.py` to fine-tune GPT-2. Once training is complete, you can chat with the model by running `chat.py`. Just point it to your model’s path, and you're ready to interact!

## Files
- **trainer.py**: Script for training GPT-2 using `transformers` and `datasets`. Loads the model, tokenizes the dataset, trains, and saves the fine-tuned model.
- **chat.py**: Script to initiate a chat interface with the fine-tuned model, enabling interactive text generation.
- **requirements.txt**: List of Python dependencies required to run the scripts.

## Usage
1. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2. **Training**:
    - Place your data in `data.txt`.
    - Run `trainer.py` to train GPT-2 on your dataset.

3. **Chatting**:
    - Run `chat.py` to interact with the fine-tuned model. Provide the model path in the `model_path` variable.

## Requirements
Ensure all dependencies in `requirements.txt` are installed.
