from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import torch
import math
from pathlib import Path
from datetime import datetime
import csv

def get_model(model_name="openlm-research/open_llama_7b"):
    return AutoModelForCausalLM.from_pretrained(model_name)

def get_tokenizer(model_name="openlm-research/open_llama_7b"):
    return AutoTokenizer.from_pretrained(model_name)

class TinyStoryDataset(Dataset):
    def __init__(self, tokenizer, tokens_required):
        self.samples = []
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        token_total = 0
        for story in dataset:
            encoded = tokenizer(
                story["text"],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=64
            )
            self.samples.append(encoded)
            token_total += len(encoded["input_ids"][0])
            if token_total >= tokens_required:
                break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {k: v.squeeze(0) for k, v in self.samples[idx].items()}

def get_dataloader(tokens: int, batch_size=4):
    tokenizer = get_tokenizer()
    dataset = TinyStoryDataset(tokenizer, tokens)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_optimizer(model, learning_rate=5e-5):
    return AdamW(model.parameters(), lr=learning_rate)

def evaluate_model(model, tokenizer=None, prompt="The quick brown fox jumps over the lazy dog."):
    model.eval()
    device = next(model.parameters()).device
    if tokenizer is None:
        tokenizer = get_tokenizer()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
    return round(perplexity, 4), round(1 / perplexity, 4)

def estimate_tops():
    return 275.0

def log_trial_result_csv(csv_path: Path, trial_data: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp", "token_count", "trial_number",
        "accuracy", "tops_used", "parameter_efficiency",
        "parameter_efficiency_loss", "parameter_perplexity"
    ]
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(trial_data)

def train_model(tokens: int, trial_number: int, output_dir: Path) -> dict:
    model = get_model()
    tokenizer = get_tokenizer()
    dataloader = get_dataloader(tokens)
    optimizer = get_optimizer(model)
    scaler = GradScaler()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    for epoch in range(3):
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            with autocast():
                outputs = model(**batch)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    perplexity, accuracy = evaluate_model(model, tokenizer)
    tops_used = estimate_tops()
    parameter_efficiency = round(accuracy / (tops_used * 7_000_000_000), 10)
    parameter_perplexity = round(perplexity / 7_000_000_000, 12)
    baseline_parameter_efficiency = 5e-9
    parameter_efficiency_loss = round(1 - (parameter_efficiency / baseline_parameter_efficiency), 6)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.txt", "w") as f:
        f.write(f"perplexity: {perplexity}\n")
        f.write(f"accuracy: {accuracy}\n")
        f.write(f"tops_used: {tops_used}\n")
        f.write(f"parameter_efficiency: {parameter_efficiency}\n")
        f.write(f"parameter_efficiency_loss: {parameter_efficiency_loss}\n")
        f.write(f"parameter_perplexity: {parameter_perplexity}\n")

    log_csv_path = output_dir.parent / "run_log.csv"
    log_trial_result_csv(log_csv_path, {
        "timestamp": datetime.now().isoformat(),
        "token_count": tokens,
        "trial_number": trial_number,
        "accuracy": accuracy,
        "tops_used": tops_used,
        "parameter_efficiency": parameter_efficiency,
        "parameter_efficiency_loss": parameter_efficiency_loss,
        "parameter_perplexity": parameter_perplexity
    })

    gc.collect()  # prompt garbage collection after each trial

    return {
        "status": "success",
        "accuracy": accuracy,
        "perplexity": perplexity,
        "tops_used": tops_used,
        "parameter_efficiency": parameter_efficiency,
        "parameter_efficiency_loss": parameter_efficiency_loss,
        "parameter_perplexity": parameter_perplexity
    }
