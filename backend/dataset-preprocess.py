import re
from datasets import Dataset 

def preprocess_text(text):
    text = re.sub(r'[&*()^%$#@~`+=-_|\:;/>.<,]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

def load_txt_file(file_path):
    with open(file_path, 'r') as file:
        # Read the file and split by new lines
        lines = file.readlines()
    
    # Return as a dataset-like structure (list of dicts, with 'label' and 'text')
    dataset = []
    for line in lines:
        label, text = line.split("\t", 1)  # Split by tab to get label and text
        dataset.append({"label": label.strip(), "text": text.strip()})

    return dataset

def preprocess_dataset(file_path):
    # Load the dataset from a text file
    dataset = load_txt_file(file_path)
    
    # Create a Hugging Face Dataset
    dataset = Dataset.from_dict({
        "label": [entry["label"] for entry in dataset],  # Extract labels
        "text": [entry["text"] for entry in dataset]     # Extract texts
    })
    
    # Apply preprocessing to the dataset's text column
    dataset = dataset.map(lambda example: {"text": preprocess_text(example["text"])})

    dataset.to_json("/Users/haydenh/Documents/GitHub/AI-Resume-Generator/backend/processed_dataset.json")
    
    return dataset

file_path = "/Users/haydenh/Documents/GitHub/AI-Resume-Generator/backend/SMS.txt"  # Replace with the path to your .txt file
processed_dataset = preprocess_dataset(file_path)

print(processed_dataset[0])  # Check the first entry
