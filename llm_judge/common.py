"""
Common data structures and utilities.
"""
from typing import List
import ast
import dataclasses
import glob
import json
import os
import re
import time
from typing import Optional
import torch
from config.model_adapter import load_model, get_conversation_template
import openai
import anthropic
from config.utils import str_to_torch_dtype
from config.model_adapter import get_conversation_template, ANTHROPIC_MODEL_LIST
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, T5ForConditionalGeneration, T5Tokenizer, AutoModelForSeq2SeqLM
from transformers import GenerationConfig
from vllm import SamplingParams

# API setting constants
API_MAX_RETRY = 1
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"

TIE_DELTA = 0.1

# Categories that need reference answers
NEED_REF_CATS = ["math", "reasoning", "coding", "arena-hard-200"]

# Extract scores from judgments
two_score_pattern = re.compile("\[\[(\d+\.?\d*),\s?(\d+\.?\d*)\]\]")
two_score_pattern_backup = re.compile("\[(\d+\.?\d*),\s?(\d+\.?\d*)\]")
one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")
chat_rating_pattern = re.compile("Rating:\s*(\d+)")
rating_pattern_backslash = re.compile("Rating: \\\\\[(\d+)\\\\\]")
additional_patterns = [
    re.compile("Option (\d+)"),  # "Option 3"
    re.compile("\[rating\]: (\d+)"),  # "[rating]: 5"
    re.compile("\[Judgment\] (\d+)"),  # "[Judgment] 5"
    re.compile("\[rating\]\n(\d+)"),  # "[rating]\n9"
    re.compile("\[rating\](\d+)\]"),  # "[rating]8]"
    re.compile("\[rating\] (\d+)"),  # "[rating] 8"
    re.compile("Rating: \[(\d+)\]"),  # "Rating: [5]"
    re.compile("\\\\\\[(\\d+)\\\\\\]")  # To match "\\[1\\]"
]

all_patterns = [two_score_pattern, two_score_pattern_backup, one_score_pattern, one_score_pattern_backup, chat_rating_pattern, rating_pattern_backslash] + additional_patterns

# Sampling temperature configs for
temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
    "arena-hard-200": 0.0,
}

reverse_model_map = {
    "model_1": "model_2",
    "model_2": "model_1",
}


@dataclasses.dataclass
class Judge:
    model_name: str
    prompt_template: dict
    ref_based: bool = False
    multi_turn: bool = False


@dataclasses.dataclass
class MatchSingle:
    question: dict
    model: str
    answer: dict
    judge: Judge
    ref_answer: dict = None
    multi_turn: bool = False


@dataclasses.dataclass
class MatchPair:
    question: dict
    model_1: str
    model_2: str
    answer_1: dict
    answer_2: dict
    judge: Judge
    ref_answer: dict = None
    multi_turn: bool = False

class ModelSingleton:
    _instance = None
    @staticmethod
    def get_instance():
        if ModelSingleton._instance is None:
            ModelSingleton()
        return ModelSingleton._instance

    def __init__(self, model_path,judge_model):
        if ModelSingleton._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            if judge_model in ["rwkv-raven-14b", "chatglm-6b"]:#单卡，正常的tokenizer
                self.model, self.tokenizer = one_load_model_and_tokenizer(model_path, judge_model)
            elif judge_model in ["fastchat-t5-3b-v1.0", "fastchat-t5-3b"]:
                device = 'cuda:0'
                self.model, self.tokenizer = load_T5model_and_tokenizer(model_path, judge_model, device)
            else:#多卡
                self.model, self.tokenizer = load_model_and_tokenizer(model_path,judge_model)
            ModelSingleton._instance = self

def one_load_model_and_tokenizer(model_path, judge_model, device='cuda:0', eval_mode=True):
    if judge_model == "chatglm-6b":
        model = AutoModel.from_pretrained(
            model_path, 
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).half().cuda()
        tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    else:
        model, tokenizer = load_model(
        model_path,
        revision="main",
        device="cuda",
        num_gpus=1,
        max_gpu_memory="40GB",
        dtype=None,
        load_8bit=False,
        cpu_offloading=False,
        debug=False,
    )

    return model, tokenizer

def load_T5model_and_tokenizer(model_path, judge_model, device):
    device = 'cuda:0'
    model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
    
    tokenizer = T5Tokenizer.from_pretrained(model_path)

    return model, tokenizer


def load_model_and_tokenizer(model_path, judge_model,eval_mode=True):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype = torch.float16,
        trust_remote_code = True,
        use_cache = True,
        device_map="auto",
        low_cpu_mem_usage=True,
    ).half()
    if eval_mode:
        model.eval()

    if judge_model == "guanaco-33b-merged":
        tokenizer = AutoTokenizer.from_pretrained(model_path, unk_token="<unk>"
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
    
    return model, tokenizer


def load_questions(question_file: str, begin: Optional[int], end: Optional[int]):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    questions = questions[begin:end]
    return questions


def load_model_answers(answer_dir: str):
    """Load model answers.

    The return value is a python dict of type:
    Dict[model_name: str -> Dict[question_id: int -> answer: dict]]
    """
    filenames = glob.glob(os.path.join(answer_dir, "*.jsonl"))
    filenames.sort()
    model_answers = {}

    for filename in filenames:
        model_name = os.path.basename(filename)[:-6]
        answer = {}
        with open(filename) as fin:
            for line in fin:
                line = json.loads(line)
                answer[line["question_id"]] = line
        model_answers[model_name] = answer

    # for model_name in model_answers:
    #     print(f"Model: {model_name}, Keys: {list(model_answers[model_name].keys())[:5]}")   

    return model_answers


def load_judge_prompts(prompt_file: str):
    """Load judge prompts.

    The return value is a python dict of type:
    Dict[judge_name: str -> dict]
    """
    prompts = {}
    with open(prompt_file) as fin:
        for line in fin:
            line = json.loads(line)
            prompts[line["name"]] = line
    return prompts


def run_judge_single(questions, answers, judge, ref_answers, multi_turns, new_judge_model, judge_model_path, llm, num_gpus : int=1, max_gpu_memory: str="16GB"):
    user_prompts = []
    full_prompts = []
    
    single_turn_template = "<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question}\n\n### Assistant A:\n{answer}\n\n<|The End of Assistant A's Conversation with User|>"
    multi_turn_template = "<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant A:\n{answer_1}\n\n### User:\n{question_2}\n\n### Assistant A:\n{answer_2}\n\n<|The End of Assistant A's Conversation with User|>"
    
    for i in range(len(questions)):
        kwargs = {}
        question = questions[i]
        answer = answers[i]
        ref_answer = ref_answers[i]
        multi_turn = multi_turns[i]

        if ref_answer is not None:
            kwargs["ref_answer_1"] = ref_answer["choices"][0]["turns"][0]
            if multi_turn:
                kwargs["ref_answer_2"] = ref_answer["choices"][0]["turns"][1]

        if multi_turn:
            user_prompt = multi_turn_template.format(
                question_1=question["turns"][0],
                question_2=question["turns"][1],
                answer_1=answer["choices"][0]["turns"][0],
                answer_2=answer["choices"][0]["turns"][1],
                **kwargs,
            )
        else:
            if 'reference' in question and question['reference']:
                kwargs["ref_answer_1"] = question['reference'][0]
                kwargs["ref_answer_2"] = ""
            else:
                kwargs["ref_answer_1"] = ""
                kwargs["ref_answer_2"] = ""

            user_prompt = judge.prompt_template["prompt_template"].format(
                question=question["turns"][0],
                answer=answer["choices"][0]["turns"][0],
                **kwargs,
            )

        user_prompts.append(user_prompt)
        system_prompt = judge.prompt_template["system_prompt"]
        full_prompt = "[Instruction]\n" + judge.prompt_template["system_prompt"] + "\n\n" + user_prompt
        full_prompts.append(full_prompt)

    if new_judge_model in ["gpt-3.5-turbo", "gpt-4"]:
        judgments = get_openai_answers(new_judge_model, user_prompts, system_prompt)
    elif new_judge_model in ANTHROPIC_MODEL_LIST:
        judgments = get_anthropic_answers(new_judge_model, user_prompts, system_prompt)
    elif new_judge_model in ["guanaco-33b-merged", "rwkv-raven-14b", "chatglm-6b", "fastchat-t5-3b-v1.0", "fastchat-t5-3b"]:
        judgments = get_other_answers(new_judge_model, full_prompts)
    else:
        judgments = get_vllm_answers(llm, judge_model_path, new_judge_model, full_prompts, num_gpus, max_gpu_memory)

    ratings = []
    re_judge_prompts = []
    error_indices = []
    re_None_judgements = []

    for i, judgment in enumerate(judgments):
        rating = -1
        if judge.prompt_template["output_format"] == "[[rating]]":
            match = None
            for pattern in all_patterns:
                match = re.search(pattern, judgment)
                if match:
                    break

            if match:
                # 处理原有的模式
                if match.re.pattern in [one_score_pattern.pattern, one_score_pattern_backup.pattern, rating_pattern_backslash.pattern]:
                    try:
                        rating = ast.literal_eval(match.group(1))
                    except:
                        rating = -1
                elif match.re.pattern == chat_rating_pattern.pattern:
                    rating = int(float(match.group(1)))
                else:
                    rating = int(float(match.group(1)))
            else:
                rating = -1
            if rating > 10 or rating < 1:
                rating = -1
            if rating == -1:
                re_user_prompt = ("Your previous response was : \n" + judgment + 
                              "\n But your output format does not meet this format:\"[[rating]]\", for example: \"Rating: [[5]]\"." +
                              "Please re-output the results")
                re_judge_prompts.append(re_user_prompt)
                error_indices.append(i) 
            re_None_judgements.append("None")
            ratings.append(rating)
        else:
            raise ValueError(
                f"invalid output format: {judge.prompt_template['output_format']}"
            )
    
    re_judgments = []
    if re_judge_prompts:
        if new_judge_model in ["gpt-3.5-turbo", "gpt-4"]:
            re_judgments = get_openai_answers(new_judge_model, re_judge_prompts, system_prompt)
        elif new_judge_model in ANTHROPIC_MODEL_LIST:
            re_judgments = get_anthropic_answers(new_judge_model, re_judge_prompts, system_prompt)
        elif new_judge_model in ["guanaco-33b-merged", "rwkv-raven-14b", "chatglm-6b", "fastchat-t5-3b-v1.0", "fastchat-t5-3b"]:
            re_judgments = get_other_answers(new_judge_model, re_judge_prompts)
        else:
            re_judgments = get_vllm_answers(llm, judge_model_path, new_judge_model, re_judge_prompts, num_gpus, max_gpu_memory)
        
        print("len(error_indices)",len(error_indices))
        for i, re_judgment in enumerate(re_judgments):
            re_rating = -1
            match = None
            for pattern in all_patterns:
                match = re.search(pattern, re_judgment)
                if match:
                    break

            if match:
                if match.re.pattern in [one_score_pattern.pattern, one_score_pattern_backup.pattern, rating_pattern_backslash.pattern]:
                    try:
                        re_rating = ast.literal_eval(match.group(1))
                    except:
                        re_rating = -1
                elif match.re.pattern == chat_rating_pattern.pattern:
                    re_rating = int(float(match.group(1)))
                else:
                    re_rating = int(float(match.group(1)))
            else:
                re_rating = -1
            if re_rating > 10 or re_rating < 1:
                re_rating = -1
            else:
                re_rating = -1
            orig_index = error_indices[i]  
            re_None_judgements[orig_index] = re_judgment  
            ratings[orig_index] = re_rating


    return ratings, user_prompts, judgments, re_None_judgements


def play_a_match_single(llm, matches: List[MatchPair], output_file: str, output_file_short: str, judge_model_path: str = None, new_judge_model: str = None, num_gpus: int = 1, max_gpu_memory: str = "16GB"):
    results = []
    judge = matches[0].judge
    questions = [match.question for match in matches]
    models= [match.model for match in matches]
    answers = [match.answer for match in matches]
    ref_answers = [match.ref_answer for match in matches]
    multi_turns = [match.multi_turn for match in matches]


    if judge.prompt_template["type"] == "single":
        scores, user_prompts, judgments, re_judgments = run_judge_single(
            questions, answers, judge, ref_answers, multi_turns, new_judge_model, judge_model_path, llm = llm, num_gpus=num_gpus, max_gpu_memory=max_gpu_memory
        )
        
        for i in range(len(matches)):
            question_id = questions[i]["question_id"]
            turn = 1 if not multi_turns[i] else 2
            result = {
                "question_id": question_id,
                "model": models[i],
                "judge": (new_judge_model, judge.prompt_template["name"]),
                "score": scores[i],
                "judgment": judgments[i],
                "re_judgment":re_judgments[i],
                "user_prompt": user_prompts[i],
                "turn": turn,
                "tstamp": time.time(),
            }
            print(
                f"question: {question_id}, turn: {turn}, model: {models[i]}, "
                f"score: {scores[i]}, "
                f"judge: {(new_judge_model, judge.prompt_template['name'])}"
            )
            results.append(result)
    else:
        raise ValueError(f"invalid judge type: {judge['type']}")

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "a") as fout:
            for result in results:
                fout.write(json.dumps(result) + "\n")

    return results

def run_judge_pair(questions, answers_a, answers_b, judge, new_judge_model, judge_model_path, ref_answers, multi_turns, llm, num_gpus : int=1, max_gpu_memory: str="16GB"):
    user_prompts = []
    full_prompts = []
    # model = judge.model_name

    for i in range(len(questions)):
        # try:
        question = questions[i]
        answer_a = answers_a[i]
        answer_b = answers_b[i]
        ref_answer = ref_answers[i]
        multi_turn = multi_turns[i]

        kwargs = {}
        if ref_answer is not None:
            kwargs["ref_answer_1"] = ref_answer["choices"][0]["turns"][0]
            if multi_turn:
                kwargs["ref_answer_2"] = ref_answer["choices"][0]["turns"][1]

        if multi_turn:
            try:
                user_prompt = judge.prompt_template["prompt_template"].format(
                    question_1=question["turns"][0],
                    question_2=question["turns"][1],
                    answer_a_1=answer_a["choices"][0]["turns"][0],
                    answer_a_2=answer_a["choices"][0]["turns"][1],
                    answer_b_1=answer_b["choices"][0]["turns"][0],
                    answer_b_2=answer_b["choices"][0]["turns"][1],
                    ref_answer_1=kwargs.get("ref_answer_1", ""),
                    ref_answer_2=kwargs.get("ref_answer_2", "")
                )
            except:
                # continue
                print(f"Error")
                # print(judge.prompt_template["prompt_template"])
        else:
            if 'reference' in question and question['reference']:
                kwargs["ref_answer_1"] = question['reference'][0]
                kwargs["ref_answer_2"] = ""
            else:
                kwargs["ref_answer_1"] = ""
                kwargs["ref_answer_2"] = ""
            
            user_prompt = judge.prompt_template["prompt_template"].format(
                question=question["turns"][0],
                answer_a=answer_a["choices"][0]["turns"][0],
                answer_b=answer_b["choices"][0]["turns"][0],
                **kwargs,
            )
        user_prompts.append(user_prompt)
        system_prompt = judge.prompt_template["system_prompt"]
        full_prompt = "[Instruction]\n" + judge.prompt_template["system_prompt"] + "\n\n" + user_prompt
        full_prompts.append(full_prompt)
    

    if new_judge_model in ["gpt-3.5-turbo", "gpt-4"]:
        judgments = get_openai_answers(new_judge_model, user_prompts, system_prompt)
    elif new_judge_model in ANTHROPIC_MODEL_LIST:
        judgments = get_anthropic_answers(new_judge_model, user_prompts, system_prompt)
    elif new_judge_model in ["guanaco-33b-merged", "rwkv-raven-14b", "chatglm-6b", "fastchat-t5-3b-v1.0", "fastchat-t5-3b"]:
        judgments = get_other_answers(new_judge_model, full_prompts)
    else:
        judgments = get_vllm_answers(llm, judge_model_path, new_judge_model, full_prompts, num_gpus, max_gpu_memory)

    winners = []
    re_judge_prompts = []
    error_indices = []
    re_None_judgements = []

    for i, judgment in enumerate(judgments):
        winner = "error"
        if re.search(r"\[\[C\]\]|\[C\]|cannot decide|unable to determine|cannot determine|not able to determine", judgment):
            winner = "tie"
        elif re.search(r"\[\[A\]\]|\[A\]", judgment):
            winner = "A"
        elif re.search(r"\[\[B\]\]|\[B\]", judgment):
            winner = "B"
        if winner == "error":
            # 构建重判的 batch 和用户提示
            re_user_prompt = ("Your previous response was : \n" + judgment + 
                              "\n But your output format does not meet this format" +
                              "\n \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, " +
                              "and \"[[C]]\" for a tie. Please re-output the results, " +
                              "If you believe you cannot determine, please answer C ")
            re_judge_prompts.append(re_user_prompt)
            error_indices.append(i)  # 记录需要重判的judgment和user_prompt的索引
        re_None_judgements.append("None")
        winners.append(winner)
    
    re_judgments = []
    if re_judge_prompts:
        if new_judge_model in ["gpt-3.5-turbo", "gpt-4"]:
            re_judgments = get_openai_answers(new_judge_model, re_judge_prompts, system_prompt)
        elif new_judge_model in ANTHROPIC_MODEL_LIST:
            re_judgments = get_anthropic_answers(new_judge_model, re_judge_prompts, system_prompt)
        elif new_judge_model in ["guanaco-33b-merged", "rwkv-raven-14b", "chatglm-6b", "fastchat-t5-3b-v1.0", "fastchat-t5-3b"]:
            re_judgments = get_other_answers(new_judge_model, re_judge_prompts)
        else:
            re_judgments = get_vllm_answers(llm, judge_model_path, new_judge_model, re_judge_prompts, num_gpus, max_gpu_memory)

        # 处理重判的结果

        print("len(error_indices)",len(error_indices))
        for i, re_judgment in enumerate(re_judgments):
            re_winner = "error"
            # print("re_judgment:",re_judgment,"\n")
            if judge.prompt_template["output_format"] == "[[A]]":
                if re.search(r"\[\[A\]\]|\[A\]", re_judgment):
                    re_winner = "A"
                elif re.search(r"\[\[B\]\]|\[B\]", re_judgment):
                    re_winner = "B"
                elif re.search(r"\[\[C\]\]|\[C\]|tie|cannot decide|unable to determine|cannot determine|not able to determine", re_judgment):
                    re_winner = "tie"

            orig_index = error_indices[i] 
            re_None_judgements[orig_index] = re_judgment  
            winners[orig_index] = re_winner 
            # user_prompts[orig_index] = re_judge_prompts[i]

    return winners, user_prompts, judgments, re_None_judgements

def get_openai_answers(new_judge_model, user_prompts, system_prompt):
    judgments = []
    for user_prompt in user_prompts:
        conv = get_conversation_template(new_judge_model)
        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], None)
        conv.set_system_message(system_prompt)
        judgment = chat_compeletion_openai(new_judge_model, conv, temperature=0,max_tokens=2048)
        judgments.append(judgment)

    return judgments

def get_anthropic_answers(new_judge_model, user_prompts, system_prompt):
    judgments = []
    for user_prompt in user_prompts:
        conv = get_conversation_template(new_judge_model)
        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], None)
        if system_prompt != "You are a helpful assistant.":
            user_prompt = "[Instruction]\n" + system_prompt + "\n\n" + user_prompt
            conv.messages[0][1] = user_prompt
        judgment = chat_compeletion_anthropic(
            new_judge_model, conv, temperature=0, max_tokens=1024
        )
        judgments.append(judgment)
        
    return judgments


def get_vllm_answers(llm, model_path, model_id, full_prompts, num_gpus, max_gpu_memory):

    judgments = []
    prompts = []
    # print(model_id,'\n\n\n','model_id')
    for j in range(len(full_prompts)):
        conv = get_conversation_template(model_id)
        qs = full_prompts[j]
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        prompts.append(prompt)
    
    sampling_params = SamplingParams(temperature=0.95, top_p=0.95, max_tokens=256)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        generated_text = output.outputs[0].text
        judgments.append(generated_text)
        # if output.outputs and hasattr(output.outputs[0], 'text'):
        #     judgments.append(output.outputs[0].text)
        # else:
        #     # Handle cases where output is not as expected
        #     judgments.append("Invalid output or format")

    return judgments

def get_other_answers(model_id, full_prompts):
    model_singleton = ModelSingleton.get_instance()
    model = model_singleton.model
    tokenizer = model_singleton.tokenizer
    
    if model_id in ["chatglm-6b", "fastchat-t5-3b-v1.0", "fastchat-t5-3b"]:
        device = 'cuda:0'
    else:
        device = model.device
    
    if model_id in ["fastchat-t5-3b-v1.0", "fastchat-t5-3b"]:
        prompts = []
        for j in range(len(full_prompts)):
            conv = get_conversation_template(model_id)
            qs = full_prompts[j]
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            prompts.append(prompt)
        
        input_ids = tokenizer(prompts, padding=True, truncation=True, return_tensors='pt').input_ids.to(device)
        attention_mask = tokenizer(prompts, padding=True, truncation=True, return_tensors='pt').attention_mask.to(device)
        temperature=0.7

        do_sample = temperature >= 1e-4
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=256,
            temperature=temperature,
            do_sample=do_sample
        )
        
        judgements = [tokenizer.decode(output_id, skip_special_tokens=True) for output_id in output_ids]
    elif model_id == "rwkv-raven-14b":
        temperature = 0.7
        judgements = []
        for j in range(len(full_prompts)):
            conv = get_conversation_template(model_id)
            qs = full_prompts[j]
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer([prompt]).input_ids

            if len(input_ids[0]) > 1024:
                input_ids[0] = input_ids[0][:1024]
            
            if temperature < 1e-4:
                do_sample = False
            else:
                do_sample = True
            try:
                output_ids = model.generate(
                    torch.as_tensor(input_ids).cuda(),
                    do_sample=do_sample,
                    temperature=temperature,
                    max_new_tokens=1024,
                )
                if model.config.is_encoder_decoder:
                    output_ids = output_ids[0]
                else:
                    output_ids = output_ids[0][len(input_ids[0]) :]

                # be consistent with the template's stop_token_ids
                if conv.stop_token_ids:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in conv.stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
                if conv.stop_str and isinstance(conv.stop_str, list):
                    stop_str_indices = sorted(
                        [
                            output.find(stop_str)
                            for stop_str in conv.stop_str
                            if output.find(stop_str) > 0
                        ]
                    )
                    if len(stop_str_indices) > 0:
                        output = output[: stop_str_indices[0]]
                elif conv.stop_str and output.find(conv.stop_str) > 0:
                    output = output[: output.find(conv.stop_str)]

                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                if conv.name == "xgen" and output.startswith("Assistant:"):
                    output = output.replace("Assistant:", "", 1).strip()
            except RuntimeError as e:
                print("ERROR question ID: ", full_prompts[j])
                output = "ERROR"
            judgements.append(output)
    else:
        prompts = []
        for j in range(len(full_prompts)):
            conv = get_conversation_template(model_id)
            qs = full_prompts[j]
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            prompts.append(prompt)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

        max_length = 1024

        encoded_inputs = tokenizer(prompts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        # encoded_inputs = tokenizer(full_prompts, padding=True, truncation=True, return_tensors='pt')
        input_ids = encoded_inputs.input_ids.to(device)
        attention_mask = encoded_inputs.attention_mask.to(device)

        judgements = []
        generate_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=256)
        for idx, user_input in enumerate(prompts):
            model_output = tokenizer.decode(generate_ids[idx][input_ids.shape[-1]:], skip_special_tokens=True)
            judgements.append(model_output)

    return judgements

#————————————————————————————————————————————————————————————————————————————————————————————————————————————-
def play_a_match_pair(llm, matches: List[MatchPair], output_file: str, output_file_short: str, judge_model_path: str = None, new_judge_model: str = None, num_gpus: int = 1, max_gpu_memory: str = "16GB"):
    results = []
    short_results = []
    judge = matches[0].judge

    questions = [match.question for match in matches]
    model_1s = [match.model_1 for match in matches]
    model_2s = [match.model_2 for match in matches]
    answer_1s = [match.answer_1 for match in matches]
    answer_2s = [match.answer_2 for match in matches]
    ref_answers = [match.ref_answer for match in matches]
    multi_turns = [match.multi_turn for match in matches]

    g1_results = run_judge_pair(
        questions, answer_1s, answer_2s, judge, new_judge_model, judge_model_path, ref_answers, multi_turns, llm=llm, num_gpus=num_gpus, max_gpu_memory=max_gpu_memory
    )
    g2_results = run_judge_pair(
        questions, answer_2s, answer_1s, judge, new_judge_model, judge_model_path, ref_answers, multi_turns, llm=llm, num_gpus=num_gpus, max_gpu_memory=max_gpu_memory
    )
    
    g1_map = {"A": "model_1", "B": "model_2"}
    g2_map = {"A": "model_2", "B": "model_1"}

    g1_winners, g1_user_prompts, g1_judgments, g1_re_judgments = g1_results
    g2_winners, g2_user_prompts, g2_judgments, g2_re_judgments = g2_results
    # print(f"g1_results: {g1_results}")
    for i in range(len(matches)):
        # g1_winner, g1_user_prompt, g1_judgment = g1_results[i]
        # g2_winner, g2_user_prompt, g2_judgment = g2_results[i]
        g1_winner = g1_winners[i]
        g1_user_prompt = g1_user_prompts[i]
        g1_judgment = g1_judgments[i]
        g2_winner = g2_winners[i]
        g2_user_prompt = g2_user_prompts[i]
        g2_judgment = g2_judgments[i]

        g1_re_judgment = g1_re_judgments[i]
        g2_re_judgment = g2_re_judgments[i]

        g1_winner = g1_map.get(g1_winner, g1_winner)
        g2_winner = g2_map.get(g2_winner, g2_winner)

        question_id = questions[i]["question_id"]
        turn = 1 if not multi_turns[i] else 2

        result = {
            "question_id": question_id,
            "model_1": model_1s[i],
            "model_2": model_2s[i],
            "g1_winner": g1_winner,
            "g2_winner": g2_winner,
            "judge": (new_judge_model, judge.prompt_template["name"]),
            "g1_user_prompt": g1_user_prompt,
            "g1_judgment": g1_judgment,
            "g1_re_judgment":g1_re_judgment,
            "g2_user_prompt": g2_user_prompt,
            "g2_judgment": g2_judgment,
            "g2_re_judgment":g2_re_judgment,
            "turn": turn,
            "tstamp": time.time(),
        }

        short_result = {
            "question_id": question_id,
            "model_1": model_1s[i],
            "model_2": model_2s[i],
            "g1_winner": g1_winner,
            "g2_winner": g2_winner,
            "judge": (new_judge_model, judge.prompt_template["name"]),
            "g1_judgment": g1_judgment,
            "g2_judgment": g2_judgment,
            "turn": turn,
            "tstamp": time.time(),
        }
        sshort_result = {
            "question_id": question_id,
            "model_1": model_1s[i],
            "model_2": model_2s[i],
            "g1_winner": g1_winner,
            "g2_winner": g2_winner,
            "judge": (new_judge_model, judge.prompt_template["name"]),
        }
        print(sshort_result,"\n")
        results.append(result)
        short_results.append(short_result)

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "a") as fout:
            for result in results:
                fout.write(json.dumps(result) + "\n")
    
    if output_file_short:
        os.makedirs(os.path.dirname(output_file_short), exist_ok=True)
        with open(output_file_short, "a") as fout:
            for short_result in short_results:
                fout.write(json.dumps(short_result) + "\n")


def chat_compeletion_openai(model, conv, temperature, max_tokens, api_dict=None):
    if api_dict is not None:
        openai.api_base = api_dict["api_base"]
        openai.api_key = api_dict["api_key"]

    for _ in range(API_MAX_RETRY):
        try:
            messages = conv.to_openai_api_messages()
            # print(messages)
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                # n=1,
                # temperature=temperature,
                # max_tokens=max_tokens,
            )
            output = response["choices"][0]["message"]["content"]
            break
        except openai.error.OpenAIError as e:
            print(type(e), e)
            output = "ERROR"
            time.sleep(API_RETRY_SLEEP)
            continue
            

    return output


def chat_compeletion_openai_azure(model, conv, temperature, max_tokens, api_dict=None):
    openai.api_type = "azure"
    openai.api_version = "2023-07-01-preview"
    if api_dict is not None:
        openai.api_base = api_dict["api_base"]
        openai.api_key = api_dict["api_key"]
    else:
        openai.api_base = os.environ["AZURE_OPENAI_ENDPOINT"]
        openai.api_key = os.environ["AZURE_OPENAI_KEY"]

    if "azure-" in model:
        model = model[6:]

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            messages = conv.to_openai_api_messages()
            response = openai.ChatCompletion.create(
                engine=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = response["choices"][0]["message"]["content"]
            break
        except openai.error.OpenAIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.error.InvalidRequestError as e:
            print(type(e), e)
            break
        except KeyError:
            print(response)
            break

    return output


def chat_compeletion_anthropic(model, conv, temperature, max_tokens):
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            c = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            prompt = conv.get_prompt()
            response = c.completions.create(
                model=model,
                prompt=prompt,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                max_tokens_to_sample=max_tokens,
                temperature=temperature,
            )
            output = response.completion
            break
        except anthropic.APIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return output.strip()


def chat_compeletion_palm(chat_state, model, conv, temperature, max_tokens):
    from fastchat.serve.api_provider import init_palm_chat

    assert model == "palm-2-chat-bison-001"

    if chat_state is None:
        chat_state = init_palm_chat("chat-bison@001")

    parameters = {
        "temperature": temperature,
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": max_tokens,
    }
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = chat_state.send_message(conv.messages[-2][1], **parameters)
            output = response.text
            break
        except Exception as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return chat_state, output


def normalize_game_key_single(gamekey, result):
    """Make the model names sorted in a game key."""
    qid, model_1, model_2 = gamekey
    if model_1 < model_2:
        return gamekey, result
    else:
        new_gamekey = (qid, model_2, model_1)
        new_result = {
            "winners": tuple(reverse_model_map.get(x, x) for x in result["winners"]),
            "g1_judgment": result["g2_judgment"],
            "g2_judgment": result["g1_judgment"],
        }
        return new_gamekey, new_result


def normalize_game_key_dict(judgment_dict):
    """Make the model names sorted in the game keys."""
    ret = {}
    for key, value in judgment_dict.items():
        new_key, new_value = normalize_game_key_single(key, value)
        ret[new_key] = new_value
    return ret


def load_pairwise_model_judgments(filename: str):
    """Load model judgments.

    The return value is a dict of type:
    Dict[judge: Tuple -> Dict[game_key: tuple -> game_result: dict]
    """
    judge_dict = {}

    for line in open(filename):
        obj = json.loads(line)
        judge = tuple(obj["judge"])
        qid, model_1, model_2 = obj["question_id"], obj["model_1"], obj["model_2"]

        if judge not in judge_dict:
            judge_dict[judge] = {}

        if "winner" in obj:
            winner = obj["winner"]
        elif "g1_winner" in obj and "g2_winner" in obj:
            g1_winner, g2_winner = obj["g1_winner"], obj["g2_winner"]
            if g1_winner == g2_winner:
                winner = g1_winner
            else:
                winner = "inconsistent"
        else:
            raise ValueError(f"Invalid keys: {list(obj.keys())}")

        gamekey = (qid, model_1, model_2)
        winners = (winner,)

        judge_dict[judge][gamekey] = {
            "winners": winners,
            "g1_judgment": obj["g1_judgment"],
            "g2_judgment": obj["g2_judgment"],
        }

    # Make the model names sorted in the game keys
    normalized = {}
    for judge, value in judge_dict.items():
        normalized[judge] = normalize_game_key_dict(value)
    return normalized


def load_single_model_judgments(filename: str):
    """Load model judgments.

    The return value is a dict of type:
    Dict[judge: Tuple -> Dict[game_key: tuple -> game_result: dict]
    """
    judge_dict = {}

    for line in open(filename):
        obj = json.loads(line)
        judge = tuple(obj["judge"])
        qid, model = obj["question_id"], obj["model"]

        if judge not in judge_dict:
            judge_dict[judge] = {}

        gamekey = (qid, model)

        judge_dict[judge][gamekey] = {
            "score": obj["score"],
            "judgment": obj["judgment"],
        }
    return judge_dict


def resolve_pairwise_judgment_dict(
    question, model_judgments_normal, model_judgments_math, multi_turn=False
):
    """Return the correct pairwise judge."""
    if multi_turn:
        if question["category"] in NEED_REF_CATS:
            return model_judgments_math[("gpt-4", "pair-math-v1-multi-turn")]
        return model_judgments_normal[("gpt-4", "pair-v2-multi-turn")]

    if question["category"] in NEED_REF_CATS:
        return model_judgments_math[("gpt-4", "pair-math-v1")]
    else:
        return model_judgments_normal[("gpt-4", "pair-v2")]


def resolve_single_judgment_dict(
    question, model_judgments_normal, model_judgments_math, multi_turn=False
):
    """Return the correct single answer grading judge."""
    if multi_turn:
        if question["category"] in NEED_REF_CATS:
            return model_judgments_math[("gpt-4", "single-math-v1-multi-turn")]
        return model_judgments_normal[("gpt-4", "single-v1-multi-turn")]

    if question["category"] in NEED_REF_CATS:
        return model_judgments_math[("gpt-4", "single-math-v1")]
    else:
        return model_judgments_normal[("gpt-4", "single-v1")]


def get_pairwise_judge_explanation(gamekey, judgment_dict):
    """Get model judge explanation."""
    try:
        qid, model_1, model_2 = gamekey
        if model_1 < model_2:
            res = judgment_dict[gamekey]
            g1_judgment, g2_judgment = res["g1_judgment"], res["g2_judgment"]
        else:
            new_gamekey = (qid, model_2, model_1)
            res = judgment_dict[new_gamekey]

            model_1, model_2 = model_1, model_2
            g1_judgment, g2_judgment = res["g2_judgment"], res["g1_judgment"]

        return (
            f"**Game 1**. **A**: {model_1}, **B**: {model_2}\n\n"
            f"**Judgment**: {g1_judgment}"
            + f"\n\n`--------------------------`\n\n"
            + f"**Game 2**. **A**: {model_2}, **B**: {model_1}\n\n"
            f"**Judgment**: {g2_judgment}"
        )
    except KeyError:
        return "N/A"


def get_single_judge_explanation(gamekey, judgment_dict):
    """Get model judge explanation."""
    try:
        qid, model = gamekey

        res = judgment_dict[gamekey]

        g1_judgment = res["judgment"]
        g1_score = res["score"]

        return (
            f"**Game 1**. **A**: {model}, **Score**: {g1_score}\n\n"
            f"**Judgment**: {g1_judgment}"
        )
    except KeyError:
        return "N/A"


def check_data(questions, model_answers, ref_answers, models, judges):
    # check model answers
    for m in models:
        assert m in model_answers, f"Missing model answer for {m}"
        m_answer = model_answers[m]
        for q in questions:
            assert (
                q["question_id"] in m_answer
            ), f"Missing model {m}'s answer to Question {q['question_id']}"
    # check ref answers
    for jg in judges.values():
        if not jg.ref_based:
            continue
        for q in questions:
            if q["category"] not in NEED_REF_CATS:
                continue
            assert (
                q["question_id"] in ref_answers[jg.model_name]
            ), f"Missing reference answer to Question {q['question_id']} for judge {jg.model_name}"


def get_model_list(answer_dir):
    file_paths = glob.glob(f"{answer_dir}/*.jsonl")
    file_names = [os.path.splitext(os.path.basename(f))[0] for f in file_paths]
    return file_names
