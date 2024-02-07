import pandas as pd
import json
import os
import random
import glob
import time
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any
import random
import string
from pathlib import Path
import shutil

def make_match_random_pairs(questions_file: str, model_answers_dir: str, output_file: str):
    models_list = ['gpt-3.5-turbo', 'guanaco-33b-merged', 'vicuna-13b-v1.5', 'WizardLM-13B-V1.2', 'vicuna-7b-v1.5','koala-13b', 
                   'gpt4all-13b-snoozy', 'mpt-7b-chat',  'oasst-sft-4-pythia-12b-epoch-3.5', 'alpaca-13b', 'fastchat-t5-3b-v1.0', 'chatglm-6b',
                   'stablelm-tuned-alpha-7b', 'dolly-v2-12b', 'llama-13b']

    # Calculate the number of digits required for the match code based on the maximum possible matches
    max_possible_matches = len(models_list) * (len(models_list) - 1) // 2
    digits = len(str(max_possible_matches))

    # Step 1: Load questions from the questions_file
    questions = {}
    with open(questions_file, 'r', encoding='utf-8') as file:
        for line in file:
            question = json.loads(line)
            questions[question["question_id"]] = {
                "content": question["turns"],
                "category": question["category"],
                "reference": question.get("reference", [])
            }

    # Step 2: Scan the model_answers_dir and read files
    model_answers = {}
    for model_name in models_list:
        model_file = Path(model_answers_dir) / f"{model_name}.jsonl"
        if model_file.exists():
            with open(model_file, 'r', encoding='utf-8') as file:
                for line in file:
                    answer = json.loads(line)
                    question_id = answer["question_id"]
                    if question_id in questions:
                        if model_name not in model_answers:
                            model_answers[model_name] = {}
                        model_answers[model_name][question_id] = answer["choices"][0]["turns"]

    # Step 3: Create random match pairs with unique_id
    matches = []
    seen_pairs = set()
    for question_id, question in questions.items():
        match_code = 1
        for model_1 in models_list:
            if question_id in model_answers[model_1]:
                other_models = [m for m in models_list if m != model_1]
                random_selected_models = random.sample(other_models, min(4, len(other_models)))
                
                for model_2 in random_selected_models:
                    if question_id in model_answers[model_2]:
                        pair = frozenset([model_1, model_2, question_id])
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            unique_id = f"{question_id}{match_code:0{digits}d}"  # Generate a fixed-length unique ID for each match
                            match = {
                                "question_id": unique_id,
                                "category": question["category"],
                                "reference": question["reference"],
                                "model_a": model_1,
                                "model_b": model_2,
                                "question_content": question["content"],
                                "conversation_a": model_answers[model_1][question_id],
                                "conversation_b": model_answers[model_2][question_id]
                            }
                            matches.append(match)
                            match_code += 1  # Increment the match code for the next pair

    # Step 4: Save the matches to the output_file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as file:
        for match in matches:
            file.write(json.dumps(match) + "\n")

def make_match_all_pairs(questions_file: str, model_answers_dir: str, output_file: str):
    models_list = ['gpt-3.5-turbo', 'guanaco-33b-merged', 'vicuna-13b-v1.5', 'WizardLM-13B-V1.2', 'vicuna-7b-v1.5','koala-13b', 
                  'gpt4all-13b-snoozy', 'mpt-7b-chat',  'oasst-sft-4-pythia-12b-epoch-3.5', 'alpaca-13b', 'fastchat-t5-3b-v1.0', 'chatglm-6b',
                  'stablelm-tuned-alpha-7b', 'dolly-v2-12b', 'llama-13b']
    

    # Calculate the number of digits required for the match code based on the maximum possible matches
    max_possible_matches = len(models_list) * (len(models_list) - 1) // 2
    digits = len(str(max_possible_matches))

    # Step 1: Load questions from the questions_file with improvements
    questions = {}
    with open(questions_file, 'r', encoding='utf-8') as file:
        for line in file:
            question = json.loads(line)
            questions[question["question_id"]] = {
                "content": question["turns"],
                "category": question["category"],
                "reference": question.get("reference", [])  # Extracting the reference if available
            }

    # Step 2: Scan the model_answers_dir and read files that are in the models_list
    model_answers = {}
    for model_name in models_list:
        model_file = Path(model_answers_dir) / f"{model_name}.jsonl"
        if model_file.exists():
            with open(model_file, 'r', encoding='utf-8') as file:
                for line in file:
                    answer = json.loads(line)
                    question_id = answer["question_id"]
                    if question_id in questions:
                        if model_name not in model_answers:
                            model_answers[model_name] = {}
                        model_answers[model_name][question_id] = answer["choices"][0]["turns"]

    # Step 3: Create match pairs with fixed-length unique codes for each pair
    matches = []
    for question_id, question in questions.items():
        match_code = 1  # Reset match code for each new question_id
        for i, model_1 in enumerate(models_list):
            for model_2 in models_list[i+1:]:
                if model_1 in model_answers and model_2 in model_answers and question_id in model_answers[model_1] and question_id in model_answers[model_2]:
                    unique_id = f"{question_id}{match_code:0{digits}d}"  # Generate a fixed-length unique ID for each match
                    match = {
                        "question_id": unique_id,
                        "category": question["category"],
                        "reference": question["reference"],  # Include the reference in the match
                        "model_a": model_1,
                        "model_b": model_2,
                        "question_content": question["content"],
                        "conversation_a": model_answers[model_1][question_id],
                        "conversation_b": model_answers[model_2][question_id]
                    }
                    matches.append(match)
                    match_code += 1  # Increment the match code for the next pair

    # Step 4: Save the matches to the output_file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as file:
        for match in matches:
            file.write(json.dumps(match) + "\n")

def assign_judges_and_save(save_folder, cleaned_jsonl, assign_num):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    cleaned_data = pd.read_json(cleaned_jsonl, lines=True)

    new_model_list = list(pd.concat([cleaned_data['model_a'], cleaned_data['model_b']]).unique())

    judge_data = {model: [] for model in new_model_list}

    for index, row in cleaned_data.iterrows():
        available_models = [model for model in new_model_list]
        judges = random.sample(available_models, assign_num) if len(available_models) >= assign_num else available_models

        for judge in judges:
            judge_data[judge].append(row.to_dict())

    for judge, data in judge_data.items():
        save_path = os.path.join(save_folder, f'{judge}.jsonl')
        pd.DataFrame(data).to_json(save_path, orient='records', lines=True)


def transform_and_save_generic_format(jsonl_data, folder_name, question_jsonl_path):
    # Transform data without adding reference
    new_data = []
    for line in jsonl_data.splitlines():
        data = json.loads(line)
        transformed = {
            "question_id": data["question_id"],
            "category": data["category"],
            "turns": [data["question_content"]]
        }
        new_data.append(json.dumps(transformed))

    question_folder = os.path.join(folder_name, 'question')
    os.makedirs(question_folder, exist_ok=True)

    # Save data to temporary file
    temp_file_path = os.path.join(question_folder, 'temp_question.jsonl')
    with open(temp_file_path, 'w') as file:
        file.write("\n".join(new_data))

    # Supplement references using the logic from the second function
    reference_jsonl(question_jsonl_path, temp_file_path)

    # Rename temp file to final file
    final_file_path = os.path.join(question_folder, 'question.jsonl')
    os.rename(temp_file_path, final_file_path)
    print("Transformation and Saving completed.")

def transform_and_save_conversation_format(jsonl_data, folder_name):
    model_data = {}
    for line in jsonl_data.splitlines():
        data = json.loads(line)
        for model_key in ["model_a", "model_b"]:
            model_id = data[model_key]
            transformed = {
                "question_id": data["question_id"],
                "answer_id": str(data["question_id"]) + ("a" if model_key == "model_a" else "b"),
                "model_id": model_id,
                "choices": [{"index": 0, "turns": data["conversation_" + model_key[-1]]}],
                "tstamp": time.time()
            }
            if model_id not in model_data:
                model_data[model_id] = []
            model_data[model_id].append(json.dumps(transformed))

    os.makedirs(folder_name, exist_ok=True)

    for model_id, data in model_data.items():
        file_name = os.path.join(folder_name, f'{model_id}.jsonl')
        with open(file_name, 'w') as file:
            file.write("\n".join(data))
    
    return list(model_data.keys())

def process_and_save_all_jsonl(original_folder, output_folder, question_path):
    count_info = []

    for file_name in os.listdir(original_folder):
        if file_name.endswith('.jsonl'):
            original_file_path = os.path.join(original_folder, file_name)

            with open(original_file_path, 'r') as file:
                original_data = file.read()

            new_folder_name = os.path.join(output_folder, file_name.replace('.jsonl', ''))
            os.makedirs(new_folder_name, exist_ok=True)

            transform_and_save_generic_format(original_data, new_folder_name, question_path)
            model_ids = transform_and_save_conversation_format(original_data, new_folder_name)

            last_folder_name = os.path.basename(new_folder_name)

            count_info.append(f"{last_folder_name}: {' '.join(model_ids)}")

    count_file_path = os.path.join(output_folder, 'count.txt')
    with open(count_file_path, 'w') as count_file:
        count_file.write("\n".join(count_info))


def reference_jsonl(question_jsonl_path, to_process_jsonl_path):
    # Step 1: Read question.jsonl and create a dictionary
    question_references = {}
    with open(question_jsonl_path, 'r', encoding='utf-8') as file:
        for line in file:
            question = json.loads(line)
            question_id = question.get("question_id")
            reference = question.get("reference")
            if reference:
                question_references[question_id] = reference

    # Step 2: Process the to_process_jsonl file
    updated_lines = []
    with open(to_process_jsonl_path, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            original_id = entry.get("question_id")
            # Remove the last three digits to find the corresponding question_id
            corresponding_id = int(original_id) // 1000
            # Check if there is a reference for this id
            if corresponding_id in question_references:
                entry["reference"] = question_references[corresponding_id]
            updated_lines.append(json.dumps(entry))

    # Step 3: Write the updated content back to the original file
    with open(to_process_jsonl_path, 'w', encoding='utf-8') as file:
        for line in updated_lines:
            file.write(line + "\n")
    print("Processing completed.")

if __name__ == "__main__":

    questions_file = './data/mt_bench/question.jsonl'
    model_answers_dir = './data/mt_bench/model_answer'
    temp_folder = './data/mt_bench/model_answer/assign'

    temp_file = temp_file = os.path.join(temp_folder, 'match.jsonl')
    judge_num = 5
    make_match_all_pairs(questions_file, model_answers_dir, temp_file)

    assign_judges_and_save(temp_folder, temp_file, judge_num)
    os.remove(temp_file)

    process_and_save_all_jsonl(temp_folder, model_answers_dir, questions_file)
    shutil.rmtree(temp_folder)