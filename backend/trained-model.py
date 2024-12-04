from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Function to load preprocessed dataset
def load_preprocessed_dataset(file_path):
    # Load the preprocessed dataset from the file you created in dataset-preprocess.py
    dataset = Dataset.from_json(file_path)  # Assuming you saved the dataset as a JSON file
    return dataset

# Load the preprocessed dataset
file_path = "/Users/haydenh/Documents/GitHub/AI-Resume-Generator/backend/processed_dataset.json"  # Replace with your actual path to the preprocessed dataset file
dataset = load_preprocessed_dataset(file_path)

# Split the dataset into train and test
dataset = dataset.train_test_split(test_size=0.2)  # 80% train, 20% test

# Convert labels to integers (if necessary)
label_map = {"ham": 0, "spam": 1}
dataset = dataset.map(lambda x: {"label": label_map[x["label"]]})

# Initialize Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set Up Training Arguments
training_args = TrainingArguments(
    output_dir="./results",          # Output directory where model checkpoints will be saved
    evaluation_strategy="epoch",     # Evaluate every epoch
    learning_rate=2e-5,              # Learning rate for training
    per_device_train_batch_size=8,   # Batch size for training
    per_device_eval_batch_size=8,    # Batch size for evaluation
    num_train_epochs=3,              # Number of training epochs
    weight_decay=0.01,               # Weight decay for optimization
)

# Set Up the Trainer
trainer = Trainer(
    model=model,                       # The model to train
    args=training_args,                # Training arguments
    train_dataset=tokenized_datasets["train"],  # Training dataset
    eval_dataset=tokenized_datasets["test"],    # Evaluation dataset
)

# Begin Model Training
trainer.train()

# Save The Trained Model
model.save_pretrained("./model")  # Save the model in a "model" directory
tokenizer.save_pretrained("./model")  # Save the tokenizer