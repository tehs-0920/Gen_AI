import csv
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

CSV_FILENAME = "gpt2_next_token_results.csv"


def load_engine(model_name: str = "openai-community/gpt2"):
    print(f"Loading {model_name} into memory... (Please wait a few seconds)")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    print(f"Model ready! Results will be saved to '{CSV_FILENAME}'.\n")
    return tokenizer, model


def save_to_csv(rows: list[list], filename: str = CSV_FILENAME):
    file_exists = os.path.isfile(filename)

    with open(filename, mode="a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "Prompt",
                "Temperature",
                "Rank",
                "Token ID",
                "Candidate Token",
                "Raw Logit",
                "Scaled Logit",
                "Probability (%)"
            ])

        writer.writerows(rows)


def analyze_prompt(tokenizer, model, prompt: str, top_k: int, temperature: float):
    print("\n" + "=" * 82)
    print(f"PROMPT: \"{prompt}\" | TOP-K: {top_k} | TEMPERATURE: {temperature}")
    print("=" * 82)

    # 1. Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"][0]

    print("\n[STAGE 1] INPUT TOKENIZATION (Text -> Subword IDs)")
    print("-" * 82)

    for pos, tid in enumerate(input_ids):
        print(f"  Pos {pos:<2} | ID: {tid.item():<6} | Token: {repr(tokenizer.decode([tid]))}")

    # 2. Forward pass to get logits for the next token
    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]

    vocab_size = next_token_logits.shape[0]

    print(f"\n[STAGE 2] RAW MODEL SCORES (Logits across all {vocab_size:,} vocabulary tokens)")
    print("-" * 82)
    print(f"  Min Logit : {next_token_logits.min().item():.4f} | Max Logit : {next_token_logits.max().item():.4f}")

    # 3. Apply Temperature & Softmax
    scaled_logits = next_token_logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)

    # 4. Top-K Extraction
    top_probs, top_ids = torch.topk(probs, top_k)

    print(f"\n[STAGE 3] TOP-{top_k} NEXT-TOKEN CANDIDATES (After Temperature & Softmax)")
    print("-" * 82)

    print(f"{'Rank':<5} | {'ID':<7} | {'Candidate':<16} | {'Raw Logit':<10} | {'Scaled':<9} | {'Prob (%)':<9} | {'Visual Bar'}")
    print("-" * 82)

    cum_prob = 0.0
    csv_rows = []

    for rank in range(top_k):
        tid = top_ids[rank].item()
        t_str = repr(tokenizer.decode([tid]))
        r_logit = next_token_logits[tid].item()
        s_logit = scaled_logits[tid].item()
        prob = top_probs[rank].item()
        cum_prob += prob

        bar = "█" * int(prob * 30)

        print(f"#{rank+1:<4} | {tid:<7} | {t_str:<16} | {r_logit:>9.4f} | {s_logit:>8.4f} | {prob*100:>7.2f}% | {bar}")

        csv_rows.append([
            prompt,
            temperature,
            rank + 1,
            tid,
            t_str,
            round(r_logit, 4),
            round(s_logit, 4),
            round(prob * 100, 2)
        ])

    save_to_csv(csv_rows)

    print("-" * 82)
    print(f"Cumulative Probability of Top-{top_k}: {cum_prob*100:.2f}% | Remaining {vocab_size - top_k:,} tokens: {(1.0 - cum_prob)*100:.2f}%")
    print(f"Saved {len(csv_rows)} rows to '{CSV_FILENAME}'")
    print("=" * 82 + "\n")


def main():
    tokenizer, model = load_engine("openai-community/gpt2")

    while True:
        prompt = input("Enter prompt (or type 'exit' to quit): ").strip()

        if prompt.lower() in ("exit", "quit", "q"):
            print(f"Exiting lab demonstration. Results saved in '{CSV_FILENAME}'.")
            break

        if not prompt:
            print("Please enter a non-empty prompt.\n")
            continue

        # Read Top-K with safe default
        k_input = input("Enter top-k candidates to display [default = 6]: ").strip()

        try:
            top_k = int(k_input) if k_input else 6
            top_k = max(1, min(top_k, 50))
        except ValueError:
            print("Invalid integer. Defaulting top-k to 6.")
            top_k = 6

        # Read Temperature with safe default
        t_input = input("Enter temperature (e.g., 0.2, 1.0, 2.0) [default = 1.0]: ").strip()

        try:
            temperature = float(t_input) if t_input else 1.0

            if temperature <= 0:
                print("Temperature must be > 0. Defaulting to 0.1.")
                temperature = 0.1

        except ValueError:
            print("Invalid number. Defaulting temperature to 1.0.")
            temperature = 1.0

        analyze_prompt(tokenizer, model, prompt, top_k, temperature)


if __name__ == "__main__":
    main()