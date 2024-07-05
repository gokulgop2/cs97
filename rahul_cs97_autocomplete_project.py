# Install the Transformers library (run this in your terminal)
# pip install transformers datasets

import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments, pipeline, set_seed

# Load JSON data
with open('recipes.json', 'r') as file:
    recipes = json.load(file)

# Convert JSON data to text format
recipes_text = ""
for recipe in recipes:
    title = recipe.get('title', 'No title')
    ingredients = recipe.get('ingredients', 'No ingredients')
    instructions = recipe.get('instructions', 'No instructions')
    recipes_text += f"Title: {title}\n"
    recipes_text += f"Ingredients: {ingredients}\n"
    recipes_text += f"Instructions: {instructions}\n\n"

# Save the formatted text to a file
with open('recipes.txt', 'w') as file:
    file.write(recipes_text)

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
model = GPT2LMHeadModel.from_pretrained('gpt2-xl')

# Prepare the dataset
def load_dataset(file_path, tokenizer):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128
    )

# Load your recipes dataset
train_dataset = load_dataset("recipes.txt", tokenizer)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    save_steps=10_000,
    save_total_limit=2,
    no_cuda=True,  # Disable GPU usage
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./fine-tuned-gpt2')
tokenizer.save_pretrained('./fine-tuned-gpt2')

# Load the fine-tuned model
model = GPT2LMHeadModel.from_pretrained('./fine-tuned-gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('./fine-tuned-gpt2')

# Create a pipeline with the fine-tuned model
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=-1)  # Ensure the CPU is used

# Set the seed for reproducibility
set_seed(42)

# Function to generate recipe
def generate_recipe(prompt):
    generated = generator(prompt, max_length=200, num_return_sequences=1, top_p=0.9)
    return generated[0]['generated_text']

# Test the function
recipe_prompt = "I made a Pizza with the ingredients and steps:"
print(generate_recipe(recipe_prompt))
