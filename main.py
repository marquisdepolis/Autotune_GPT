import time
import sys
import re
import openai
import json
import os
import subprocess
from dotenv import load_dotenv

load_dotenv()


def read_model_name_from_file():
    with open("current_model.txt", "r") as f:
        return f.read().strip()


def generate_text(prompt, model):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.5
    )

    return response.choices[0].text.strip()


def has_ten_lines(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    return len(lines) >= 10


def save_jsonl(prompt, completion, file_path):
    data = {"prompt": prompt, "completion": completion}

    with open(file_path, "a") as f:
        f.write(json.dumps(data) + "\n")


def write_model_name_to_file(model_name):
    with open("current_model.txt", "w") as f:
        f.write(model_name)


def prepare_data(file_path):
    subprocess.run(
        ["openai", "tools", "fine_tunes.prepare_data", "-f", file_path]
    )


def create_fine_tuned_model(jsonl_file_path):
    result = subprocess.run(
        ['openai', 'api', 'fine_tunes.create',
            '-t', jsonl_file_path, '-m', 'davinci'],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)

    fine_tune_id = re.search(
        r'Created fine-tune: (\S+)', result.stdout).group(1)
    print(f"Fine-tune ID: {fine_tune_id}")

    # Wait for the fine-tuning to complete
    while True:
        time.sleep(60)
        status_result = subprocess.run(
            ['openai', 'api', 'fine_tunes.get', '-i', fine_tune_id],
            capture_output=True,
            text=True,
        )
        status_data = json.loads(status_result.stdout)
        status = status_data["status"]
        print(f"Fine-tuning status: {status}")

        if status == "succeeded":
            fine_tuned_model = status_data["fine_tuned_model"]
            with open("current_model.txt", "w") as f:
                f.write(fine_tuned_model)
            print(f"Fine-tuned model saved: {fine_tuned_model}")
            break
        elif status in {"failed", "canceled"}:
            print(f"Fine-tuning failed or canceled: {status}")
            sys.exit(1)


def main_function():
    jsonl_file_path = "data.jsonl"
    country_prompts = [
        "What is the capital of France?",
        "What is the capital of Germany?",
        "What is the capital of Italy?",
        "What is the capital of Spain?",
        "What is the capital of Portugal?",
        "What is the capital of the United Kingdom?",
        "What is the capital of Russia?",
        "What is the capital of Japan?",
        "What is the capital of Australia?",
        "What is the capital of Brazil?",
        "What is the capital of Guinea Bissau?"
    ]

    for prompt in country_prompts:
        # Call GPT using the generate_text function
        current_model = read_model_name_from_file()
        completion = generate_text(prompt, current_model)
        save_jsonl(prompt, completion, jsonl_file_path)

    if has_ten_lines(jsonl_file_path):
        print("Preparing data and starting fine-tuning...")
        prepare_data(jsonl_file_path)
        create_fine_tuned_model(jsonl_file_path)
    else:
        print("Not enough lines in the JSONL file for fine-tuning.")


if __name__ == "__main__":
    main_function()
