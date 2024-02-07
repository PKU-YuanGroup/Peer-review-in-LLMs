from vllm import LLM, SamplingParams
import argparse
import jsonlines
import torch
from concurrent.futures import ThreadPoolExecutor
import json
import os
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from config.model_adapter import load_model, get_conversation_template
from config.utils import str_to_torch_dtype
from config.model_adapter import get_conversation_template, ANTHROPIC_MODEL_LIST
from common import (
    load_questions,
    load_model_answers,
    load_judge_prompts,
    check_data,
    play_a_match_pair,
    play_a_match_single,
    get_model_list,
    Judge,
    MatchPair,
    MatchSingle,
    NEED_REF_CATS,
)
from common import ModelSingleton

def make_match(
    questions,
    models,
    model_answers,
    judge,
    baseline_model,
    existing_matches_file=None,
    ref_answers=None,
    multi_turn=False,
):
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for i in range(len(models)):
            q_id = q["question_id"]
            m_1 = models[i]
            m_2 = baseline_model
            if m_1 == m_2:
                continue
            a_1 = model_answers[m_1][q_id]
            a_2 = model_answers[baseline_model][q_id]
            if ref_answers is not None:
                ref = ref_answers[judge.model_name][q_id]
                match = MatchPair(
                    dict(q),
                    m_1,
                    m_2,
                    a_1,
                    a_2,
                    judge,
                    ref_answer=ref,
                    multi_turn=multi_turn,
                )
            else:
                match = MatchPair(
                    dict(q), m_1, m_2, a_1, a_2, judge, multi_turn=multi_turn
                )
            matches.append(match)
    return matches


def Chatbot_make_match_all_pairs(
    questions,
    models,
    model_answers,
    judge,
    baseline_model=None,
    existing_matches_file=None,
    ref_answers=None,
    output_file=None,       
    multi_turn=False,
):

    existing_matches = set()
    if existing_matches_file:
        with open(existing_matches_file, "r") as file:
            for line in file:
                data = json.loads(line)
                int_var = str(data["question_id"])
                match_key = (int_var, data["model_1"], data["model_2"])
                # print(type(int_var))
                existing_matches.add(match_key)

    print(f"Loaded {len(existing_matches)} existing matches.")

    skipped_matches_count = 0


    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue

        for i in range(len(models)):
            m_1 = models[i]
            for j in range(i + 1, len(models)):
                m_2 = models[j]
                q_id = q["question_id"]
                # print(type(q_id))
                if q_id not in model_answers.get(m_1, {}) or q_id not in model_answers.get(m_2, {}):
                    continue
                
                # print(type(q_id))
                if (q_id, m_1, m_2) in existing_matches:
                    skipped_matches_count += 1
                    continue


                if (q_id, m_2, m_1) in existing_matches:
                    skipped_matches_count += 1
                    continue

                a_1 = model_answers[m_1][q_id]
                a_2 = model_answers[m_2][q_id]

                if ref_answers is not None and judge.model_name in ref_answers and q_id in ref_answers[judge.model_name]:
                    ref = ref_answers[judge.model_name][q_id]
                    match = MatchPair(
                        dict(q),
                        m_1,
                        m_2,
                        a_1,
                        a_2,
                        judge,
                        ref_answer=ref,
                        multi_turn=multi_turn,
                    )
                else:
                    match = MatchPair(
                        dict(q), m_1, m_2, a_1, a_2, judge, multi_turn=multi_turn
                    )
                matches.append(match)

    print(f"Successfully skipped {skipped_matches_count} existing matches.")

    return matches


def make_match_single( 
    questions,
    models,
    model_answers,
    judge,
    baseline_model=None,
    existing_matches_file=None,
    ref_answers=None,
    multi_turn=False,
):
    matches = []
    skipped_count = 0
    existing_matches = {}
    if existing_matches_file:
        with open(existing_matches_file, 'r') as file:
            for line in file:
                match = json.loads(line)
                key = (match['question_id'], match['model'])
                existing_matches[key] = match
        print(f"Loaded {len(existing_matches)} existing matches.")

    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for i in range(len(models)):
            q_id = q["question_id"]
            m = models[i]

            if (q_id, m) in existing_matches:
                skipped_count += 1
                continue  

            a = model_answers[m][q_id]
            if ref_answers is not None:
                ref = ref_answers[judge.model_name][q_id]
                matches.append(
                    MatchSingle(
                        dict(q), m, a, judge, ref_answer=ref, multi_turn=multi_turn
                    )
                )
            else:
                matches.append(MatchSingle(dict(q), m, a, judge, multi_turn=multi_turn))

    print(f"Skipped {skipped_count} existing matches.")
    return matches


def make_chatbot_match_single( 
    questions,
    models,
    model_answers,
    judge,
    baseline_model=None,
    existing_matches_file=None,
    ref_answers=None,
    multi_turn=False,
):
    matches = []
    skipped_count = 0

    existing_matches = {}
    if existing_matches_file:
        with open(existing_matches_file, 'r') as file:
            for line in file:
                match = json.loads(line)
                key = (match['question_id'], match['model'])
                existing_matches[key] = match
        print(f"Loaded {len(existing_matches)} existing matches.")

    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        q_id = q["question_id"]
        for i in range(len(models)):
            m = models[i]

            # Check if the model provided an answer for this question
            if q_id not in model_answers[m]:
                continue

            if (q_id, m) in existing_matches:
                skipped_count += 1
                continue  

            a = model_answers[m][q_id]
            if ref_answers is not None:
                ref = ref_answers[judge.model_name][q_id]
                matches.append(
                    MatchSingle(
                        dict(q), m, a, judge, ref_answer=ref, multi_turn=multi_turn
                    )
                )
            else:
                matches.append(MatchSingle(dict(q), m, a, judge, multi_turn=multi_turn))

    print(f"Skipped {skipped_count} existing matches.")
    return matches


def make_judge_pairwise(judge_model, judge_prompts):
    judges = {}
    judges["default"] = Judge(judge_model, judge_prompts["pair-v2"])
    judges["math"] = Judge(judge_model, judge_prompts["pair-math-v1"], ref_based=True)
    judges["default-mt"] = Judge(
        judge_model, judge_prompts["pair-v2-multi-turn"], multi_turn=True
    )
    judges["math-mt"] = Judge(
        judge_model,
        judge_prompts["pair-math-v1-multi-turn"],
        ref_based=True,
        multi_turn=True,
    )
    return judges


def make_judge_single(judge_model, judge_prompts):
    judges = {}
    judges["default"] = Judge(judge_model, judge_prompts["single-v1"])
    judges["math"] = Judge(judge_model, judge_prompts["single-math-v1"], ref_based=True)
    judges["default-mt"] = Judge(
        judge_model, judge_prompts["single-v1-multi-turn"], multi_turn=True
    )
    judges["math-mt"] = Judge(
        judge_model,
        judge_prompts["single-math-v1-multi-turn"],
        ref_based=True,
        multi_turn=True,
    )
    return judges

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--judge-file",
        type=str,
        default="data/judge_prompts.jsonl",
        help="The file of judge prompts.",
    )
    parser.add_argument("--judge-model", type=str, default="gpt-4")

    parser.add_argument("--new-judge-model", type=str, default="gpt-4")

    parser.add_argument("--baseline-model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["pairwise-baseline", "pairwise-all", "single"],
        help=(
            "Evaluation mode. "
            "`pairwise-baseline` runs pairwise comparision against a baseline. "
            "`pairwise-all` runs pairwise comparision between all pairs. "
            "`single` runs single answer grading."
        ),
    )
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1, 
        help="Number of GPUs to use."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        default="35GB",  
        help="Maximum GPU memory to use."
    )
    parser.add_argument(
        "--judge-model-path",
        type=str,
        default=None,
        help="Path to the judge model file."
    )
    
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    parser.add_argument(
        "--first-n", type=int, help="A debug option. Only run the first `n` judgments."
    )
    parser.add_argument(
        "--batch-size", type=int, help="The batch size for processing."
    )

    parser.add_argument(
        "--existing-judge-file",
        type=str,
        default=None,
        help="Path to the existing judge JSONL file, help with intermittent reconnection"
    )

    parser.add_argument(
        "--data-split",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    question_file = f"data/{args.bench_name}/model_answer/{args.new_judge_model}/question/question.jsonl"
    answer_dir = f"data/{args.bench_name}/model_answer/{args.new_judge_model}"
    ref_answer_dir = f"data/{args.bench_name}/reference_answer"
    # Load questions
    questions = load_questions(question_file, None, None)
    # Load answers
    model_answers = load_model_answers(answer_dir)
    ref_answers = load_model_answers(ref_answer_dir)

 
    judge_prompts = load_judge_prompts("data/judge_prompts.jsonl")

    if args.first_n:
        questions = questions[: args.first_n]

    if args.model_list is None:
        models = get_model_list(answer_dir)
    else:
        models = args.model_list

    if args.mode == "single":
        judges = make_judge_single(args.judge_model, judge_prompts)
        play_a_match_func = play_a_match_single
        output_file = (
            f"data/{args.bench_name}/model_judgment/{args.new_judge_model}_single.jsonl"
        )
        output_file_short = (
                f"data/{args.bench_name}/model_judgment/{args.new_judge_model}_single_short.jsonl"
        )
        if args.bench_name == 'Chatbot_Arena_bench':
            make_match_func = make_chatbot_match_single
        else:
            make_match_func = make_match_single
        baseline_model = None
    else:
        judges = make_judge_pairwise(args.judge_model, judge_prompts)
        play_a_match_func = play_a_match_pair
        output_file = (
            f"data/{args.bench_name}/model_judgment/{args.new_judge_model}_pair.jsonl"
        )
        output_file_short = (
            f"data/{args.bench_name}/model_judgment/{args.new_judge_model}_pair_short.jsonl"
        )

        if args.mode == "pairwise-all":
            make_match_func = Chatbot_make_match_all_pairs
            baseline_model = None
        else:
            make_match_func = make_match
            baseline_model = args.baseline_model

    question_math = [q for q in questions if q["category"] in NEED_REF_CATS]
    question_default = [q for q in questions if q["category"] not in NEED_REF_CATS]

    existing_matches_file=args.existing_judge_file

    # Make matches
    matches = []
    matches += make_match_func(
        question_default, models,  model_answers, judges["default"], baseline_model,existing_matches_file,output_file,
    )
    matches += make_match_func(
        question_math,
        models,
        model_answers,
        judges["math"],
        baseline_model,
        existing_matches_file,
        ref_answers,
        output_file,
    )
    matches += make_match_func(
        question_default,
        models,
        model_answers,
        judges["default-mt"],
        existing_matches_file,
        baseline_model,
        output_file,
        multi_turn=True,
    )
    matches += make_match_func(
        question_math,
        models,
        model_answers,
        judges["math-mt"],
        existing_matches_file,
        baseline_model,
        ref_answers,
        output_file,
        multi_turn=True,
    )
    
    match_stat = {}
    match_stat["bench_name"] = args.bench_name
    match_stat["mode"] = args.mode

    match_stat["judge"] = args.new_judge_model
    match_stat["judge_model_path"] = args.judge_model_path  
    match_stat["baseline"] = baseline_model
    match_stat["model_list"] = models
    match_stat["num_gpus"] = args.num_gpus
    match_stat["max_gpu_memory"] = args.max_gpu_memory
    match_stat["batch_size"] = args.batch_size
    match_stat["total_num_questions"] = len(questions)
    match_stat["total_num_matches"] = len(matches)
    match_stat["output_path"] = output_file


    # Show match stats and prompt enter to continue
    print("Stats:")
    print(json.dumps(match_stat, indent=4))
    input("Press Enter to confirm...")


    if args.new_judge_model in ["guanaco-33b-merged", "rwkv-raven-14b", "chatglm-6b","fastchat-t5-3b-v1.0","fastchat-t5-3b"]:
        llm = None
        model_singleton = ModelSingleton(model_path=args.judge_model_path, judge_model=args.new_judge_model)
    elif args.new_judge_model not in ["gpt-3.5-turbo", "gpt-4"] and args.new_judge_model not in ANTHROPIC_MODEL_LIST:
        llm = LLM(model=args.judge_model_path, tensor_parallel_size=args.num_gpus, trust_remote_code=True)
    else:
        llm = None

    batch_size = args.batch_size
    match_batches = [matches[i:i + batch_size] for i in range(0, len(matches), batch_size)]
    if args.parallel == 1:
        for batch in tqdm(match_batches):
            play_a_match_func(llm, batch, output_file=output_file, output_file_short=output_file_short, judge_model_path=args.judge_model_path, new_judge_model = args.new_judge_model, num_gpus=args.num_gpus, max_gpu_memory= args.max_gpu_memory)
    else:
        with ThreadPoolExecutor(args.parallel) as executor:
            for _ in tqdm(
                executor.map(lambda batch: play_a_match_func(llm, batch, output_file=output_file, output_file_short=output_file_short, judge_model_path=args.judge_model_path, new_judge_model = args.new_judge_model, num_gpus=args.num_gpus, max_gpu_memory= args.max_gpu_memory), match_batches), 
                total=len(match_batches)
            ):
                pass