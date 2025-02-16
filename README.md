<p align="center">
    <img src="./assets/cute.png" width="600" style="margin-bottom: 0.2;"/>

<h2 align="center"> <a href="https://arxiv.org/pdf/2402.01830">[ICLR'25] PiCO: Peer Review in LLMs based on the Consistency Optimization</a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2>


[![arXiv](https://img.shields.io/badge/Arxiv-2402.01830-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2402.01830) 
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/PKU-YuanGroup/Peer-review-in-LLMs/blob/main/LICENSE) 


## 🤗 Brief Intro

<p align="center">
<img src="./assets/law.jpg" width=100%>
</p>

Existing large language models (LLMs) evaluation methods typically focus on testing the performance on some closed-environment and domain-specific benchmarks with human annotations. In this paper, we explore a novel **unsupervised evaluation direction**, utilizing **_peer-review_** mechanisms to measure LLMs automatically without any human feedback.
In this setting, both open-source and closed-source LLMs lie in the same environment, capable of answering unlabeled questions and evaluating each other, where each LLM’s response score is jointly determined by other anonymous ones. 
During this process, we found that those answers that are more recognized by other ``reviewers'' (models) usually come from LLMs with stronger abilities, while these models can also evaluate others' answers more accurately. 
We formalize it as a **_consistency assumption_**, _i.e._, the ability and score of the model usually have consistency. 
We exploit this to optimize each model's confidence, thereby re-ranking the LLMs to be closer to human rankings.
We perform experiments on multiple datasets with standard rank-based metrics, validating the effectiveness of the proposed approach.

## 🤗 The Pipeline of PiCO
Our PiCO framework integrates peer review with LLMs for response evaluation. It unfolds in two phases: first, LLMs generate and peer-review responses to questions, ranking them by confidence. Next, we refine these weights through consistency optimization, aiming to **minimize the system's entropy**, thereby achieving the final ranking
<p align="center">
<img src="./assets/peer-review-cute.png" width=100%>
</p>

## 🚀 Main Results

Our experimental results, derived from tests conducted on the Chatbot Arena, MT-Bench, and Alpaca-Farm datasets. Across all three datasets, our Peer Review method (Ours) **outperformed the other three settings on all metrics**, demonstrating its robustness and generalizability across different datasets.

<p align="center">
<img src="./assets/main_results.png" width=95%>
</p>

## 🛠️ Quick Start

### 🔧 Requirements and Installation
Clone this repo and install the requirements.
```bash
$ git clone https://github.com/PKU-YuanGroup/Peer-review-in-LLMs
$ cd Peer-review-in-LLMs
$ pip install --upgrade pip
$ pip install -r requirements.txt
```
### 📝 Download Data (if required)
You can directly use our complete dataset by downloading it from this link: 
```bash
https://drive.google.com/drive/folders/1c2tDhaGiuwrtwja05EFVMZCweXey86sM?usp=sharing. 
```

After downloading, place the 'mt_bench' folder into the 'data' folder. Alternatively, you can simply run the Peer review code in its entirety.

### 🖥️ Demo
After downloading the data, you can directly run main.py to experience the optimization process of Consistency Optimization using the review results from 15 models on the mt_bench dataset.
```bash
$ python main.py
```

### ✒️ LLMs response, assign and judge

This code is executed within the 'llm_judge' folder.
```bash
$ cd Peer-review-in-LLMs/llm_judge
```

#### Retrieve LLM Responses

Generate model responses
```bash
$ python gen_model_answer.py \
--model-path [/path/to/model] \
--model-id [MODEL-ID] \
--bench-name mt_bench 
```
If using proprietary models such as GPT-3.5 or GPT-4 as reviewers, you need to enter the key first.
```bash
$ export OPENAI_API_KEY=XXXXXX  # set the OpenAI API key
$ python gen_model_answer.py \
--model-id [MODEL-ID] \
--bench-name mt_bench 
```
Arguments:
  - `[MODEL-PATH]` is the path to the weights, which can be a local folder or a Hugging Face repo ID.
  - `[MODEL-ID]` is a name you give to the model.
  - `[BENCH-NAME]` is the name the dataset you use

e.g.,
```
python gen_model_answer.py --model-path PKU-YuanGroup/vicuna-7b-v1.5 --model-id vicuna-7b-v1.5 --bench-name mt_bench 
```

#### Assign Reviewers to LLM Battles
After retrieving all responses from the models to be evaluated, pair the responses and assign Reviewers.

```bash
$ python assign_judge.py
```

####  Review the Battle Pairs
After assigning reviewers, review the battle pairs and output the judgment.
```bash
$ python gen_judgment.py \
--mode pairwise-all \
--new-judge-model [MODEL-ID] \
--model-list [LIST-OF-MODEL-ID]\ 
--batch-size [BATCH-SIZE] \
--bench-name mt_bench \
--judge-model-path [/path/to/model] 
```
If using proprietary models such as GPT-3.5 or GPT-4 as reviewers, you need to enter the key first.
```bash
$ export OPENAI_API_KEY=XXXXXX  # set the OpenAI API key
$ python gen_judgment.py \
--mode pairwise-all \
--new-judge-model [MODEL-ID] \
--model-list [LIST-OF-MODEL-ID]\ 
--bench-name mt_bench 
```

Arguments:
  - `[MODE]` is the method of conducting reviews.
  - `[MODEL-ID]` is a name you give to the model.
  - `[LIST-OF-MODEL-ID]` is the list of models being reviewed.
  - `[BATCH-SIZE]` is the batch size for processing at one time.
  - `[BENCH-NAME]` is the name the dataset you use
  - `[JUDGE-MODEL-PATH]` is the path to the weights.
e.g.,
```
$ python gen_judgment.py \
--mode pairwise-all \
--new-judge-model vicuna-7b-v1.5 \
--model-list WizardLM-13B-V1.2 gpt4all-13b-snoozy oasst-sft-4-pythia-12b-epoch-3.5 vicuna-7b-v1.5 chatglm-6b guanaco-33b-merged stablelm-tuned-alpha-7b gpt-3.5-turbo koala-13b llama-13b fastchat-t5-3b-v1.0 dolly-v2-12b vicuna-13b-v1.5 alpaca-13b mpt-7b-chat \ 
--batch-size 512 \
--bench-name mt_bench \
--judge-model-path PKU-YuanGroup/vicuna-7b-v1.5
```

### ⚖️ Consistency Optimization
The code for the Consistency Optimization stage is executed within the 'Peer-review-in-LLMs/con_optimization' folder, and the results are saved in the 'log' folder.

```bash
$ python main_ablation.py \
--baseline [BASELINE] \
--mode [MODE] \
--epoch [EPOCH]

```

Arguments:
  - `[BASELINE]` determines whether to run the baseline. If set to 0, it runs Peer-review (Ours).
  - `[MODE]` selects the baseline mode, with three options: Reversed, Uniform, and Order.
  - `[EPOCH]` is the number of epochs for conducting Peer-review (Ours)

## ✏️ Citation
If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil:.

```BibTeX
@misc{ning2024peerreviewinllms,
      title={Peer-review-in-LLMs: Automatic Evaluation Method for LLMs in Open-environment}, 
      author={Kun-Peng Ning and Shuo Yang and Yu-Yang Liu and Jia-Yu Yao and Zhen-Hui Liu and Yu Wang and Ming Pang and Li Yuan},
      year={2024},
      eprint={2402.01830},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
