import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict

# Load the dataset
df = pd.read_csv('dataset.csv')

# Define your columns and label mappings
label_columns = ['Preference Profile', 'Buyer Profile', 'Buyer Argument Profile', 'Seller Argument Profile', 'Negotiation Strategy']

# Define mappings (same as before)
preference_profile_mapping = {
    "Collector's Haven": 0, 'TranquilEscape': 1, 'SightTour': 2, 'Action Agent': 3, 
    'Adrenaline Rush': 4, 'Sandy Serenity': 5, 'Active Pursuits': 6, 'Nature Wanderer': 7, 
    'Cultural Odyssey': 8, 'Nautical Adventure': 9, 'Vibrant Nightlife': 10
}
buyer_profile_mapping = {
    'Budget-conscious buyer': 0, 'Quality-conscious buyer': 1, 'Balanced buyer': 2
}
buyer_argument_profile_mapping = {
    'Agreeable': 0, 'Disagreeable': 1
}
seller_argument_profile_mapping = {
    'Open-minded seller': 0, 'Argumentative seller': 1
}
negotiation_strategy_mapping = {
    'Boulware(0.3)': 0, 'Boulware(0.6)': 1, 'Conceder(1.2)': 2
}

# Map the labels to integer values in the dataframe
df['Preference Profile'] = df['Preference Profile'].map(preference_profile_mapping).astype(float)
df['Buyer Profile'] = df['Buyer Profile'].map(buyer_profile_mapping).astype(float)
df['Buyer Argument Profile'] = df['Buyer Argument Profile'].map(buyer_argument_profile_mapping).astype(float)
df['Seller Argument Profile'] = df['Seller Argument Profile'].map(seller_argument_profile_mapping).astype(float)
df['Negotiation Strategy'] = df['Negotiation Strategy'].map(negotiation_strategy_mapping).astype(float)

# Group the data by conversation ID and concatenate the utterances for each conversation
grouped_df = df.groupby('conv_id').agg({
    'Utterance': lambda x: ' '.join(x),
    'Preference Profile': 'first',  # Assume same label per conversation
    'Buyer Profile': 'first',
    'Buyer Argument Profile': 'first',
    'Seller Argument Profile': 'first',
    'Negotiation Strategy': 'first'
}).reset_index()

# Split into training and test sets
train_df, test_df = train_test_split(grouped_df, test_size=0.2, random_state=42)

# Select only 100 samples for training and 10 samples for testing
train_df = train_df.sample(n=100, random_state=42)
test_df = test_df.sample(n=10, random_state=42)

# Convert DataFrame to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Define the tokenizer and model
llm_name = input("Enter the LLM name (e.g., 'meta-llama/Llama-2-7b-hf'): ")
output_file_name = input("Enter the output file name (e.g., 'output_results.csv'): ")
tokenizer = AutoTokenizer.from_pretrained(llm_name)

# Check if tokenizer has a pad_token, if not, set it to eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['Utterance'], padding='max_length', truncation=True)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Load the pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(
    llm_name, 
    num_labels=5,  # We have 5 labels to predict
    problem_type="multi_label_classification"  # Ensure we set this for multi-label classification
)

# Define the labels we want to predict
def compute_metrics(pred):
    # Get the predictions and labels
    logits, labels = pred
    preds = (torch.sigmoid(logits) > 0.5).float()  # Get boolean predictions
    return {
        'accuracy': (preds == labels).float().mean().item()
    }

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    save_steps=500
)

# Prepare Trainer API
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")

# Now, let's use the fine-tuned model to make predictions on the test set
# Reload the fine-tuned model
model = AutoModelForSequenceClassification.from_pretrained("./finetuned_model")

# Function to make predictions for the test dataset
def predict_test_set(test_df, model):
    tokenized_test = tokenizer(test_df['Utterance'].tolist(), truncation=True, padding=True, return_tensors='pt')
    # Move the tensor to the appropriate device (GPU or CPU)
    tokenized_test = {key: val.to(model.device) for key, val in tokenized_test.items()}  # Ensures tensors are on the same device as model
    outputs = model(**tokenized_test)
    # Apply sigmoid and threshold
    predictions = (torch.sigmoid(outputs.logits) > 0.5).float()  # Get boolean predictions
    return predictions.int().tolist()  # Convert boolean tensor to integer

# Make predictions on the test set
predictions = predict_test_set(test_df, model)

# Add predictions to the test dataframe
test_df['Predicted Profiles'] = predictions

# Save the test predictions to a CSV file
test_df.to_csv(output_file_name, index=False)

print(f"Predicted profiles for the test set saved to {output_file_name}")
