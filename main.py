import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset

def load_data(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Erreur: Le fichier {file_path} n'a pas été trouvé.")
        return None
    except json.JSONDecodeError:
        print(f"Erreur: Le fichier {file_path} n'est pas un JSON valide.")
        return None

def validate_data(data):
    for i, example in enumerate(data):
        if 'instruction' not in example:
            print(f"Erreur: L'entrée {i} ne contient pas de clé 'instruction'.")
            return False
        if 'input' not in example:
            print(f"Erreur: L'entrée {i} ne contient pas de clé 'input'.")
            return False
        if 'output' not in example:
            print(f"Erreur: L'entrée {i} ne contient pas de clé 'output'.")
            return False
    return True

# Charger les données à partir d'un fichier JSON
data = load_data('medical_data.json')
if data is None:
    exit(1)

if not validate_data(data):
    exit(1)

# Formater les données
def format_example(example):
    text = f"Instruction: {example['instruction']}\nInput: {example['input']}\nOutput: {example['output']}"
    return {'text': text}

formatted_data = [format_example(example) for example in data]
dataset = Dataset.from_list(formatted_data)

# Charger le tokeniseur et le modèle
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Tokeniser les données
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Configurer l'entraînement
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

# Entraîner le modèle
trainer.train()

# Sauvegarder le modèle
model.save_pretrained('./medical_gpt')
tokenizer.save_pretrained('./medical_gpt')

# Fonction pour générer des réponses
def generate_response(instruction, input_text):
    prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=500, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extraire seulement la partie après "Output:"
    response = response.split("Output:")[1].strip()
    return response

# Exemple d'utilisation
instruction = "If you are a doctor, please answer the medical questions based on the patient's description."
input_text = "I woke up this morning feeling the whole room is spinning when i was sitting down..."
print(generate_response(instruction, input_text))
