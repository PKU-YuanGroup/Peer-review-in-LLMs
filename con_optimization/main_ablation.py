import os
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from scipy.stats import spearmanr
import numpy as np
from itertools import permutations
from utils import *
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import logging
import argparse

seed = 4
type = 'rank' if False else 'default'
random.seed(seed)
torch.manual_seed(seed)


def evaluate(w, G):
    w_np = w.detach().numpy()
    G_np = G.detach().numpy()
    rho, p_value = spearmanr(w_np, G_np)
    return -torch.tensor(rho, requires_grad=True)


def pearson_correlation(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    diff_x = x - mean_x
    diff_y = y - mean_y
    numerator = torch.sum(diff_x * diff_y)
    denominator = torch.sqrt(torch.sum(diff_x**2) * torch.sum(diff_y**2))
    correlation = numerator / denominator
    return 1 - correlation

def cal_score(battles_df, w, n):
    init_G = Variable(torch.zeros(n))
    init_P = Variable(torch.zeros(n))
    log_dict = {ele: {'battle': 0, 'judge': 0} for ele in model_list}
    # win:1, tie: 0.5, lose: 0
    for index, row in battles_df.iterrows():
        # G_1, G_2
        winner_list = [row['g1_winner'], row['g2_winner']]
        if winner_list.count('model_1') > winner_list.count('model_2'):
          G_1, G_2 = 1, 0
        elif winner_list.count('model_1') < winner_list.count('model_2'):
          G_1, G_2 = 0, 1
        elif winner_list.count('model_1') == winner_list.count('model_2'):
          G_1, G_2 = 0.5, 0.5

        if row['model_1'] not in model_list or row['model_2'] not in model_list:
            continue

        init_G[model_list.index(row['model_1'])] += G_1 * w[model_list.index(row['judge'][0])]
        init_G[model_list.index(row['model_2'])] += G_2 * w[model_list.index(row['judge'][0])]
        error_penalize = 0.05 if winner_list.count('error') > 0 else 0
        init_P[model_list.index(row['judge'][0])] += error_penalize * w[model_list.index(row['judge'][0])]
        # init_G[model_list.index(row['model_1'])] += G_1
        # init_G[model_list.index(row['model_2'])] += G_2

        log_dict[row['model_1']]['battle'] += 1
        log_dict[row['model_2']]['battle'] += 1
        log_dict[row['judge'][0]]['judge'] += 1

    for i in range(len(model_list)):
        init_G[i] /= log_dict[model_list[i]]['battle']
        init_P[i] /= log_dict[model_list[i]]['judge']
        init_G[i] -= init_P[i]
    return init_G

def cal_score_rank(battles_df, w, n):
  init_G = Variable(torch.zeros(n))
  init_P = Variable(torch.zeros(n))
  log_dict = {ele: {'battle': 0, 'judge': 0} for ele in model_list}

  rank = torch.argsort(w, descending=True).tolist()
  rank_dict = {model_list[rank[i]]: i for i in range(len(rank))}
  
  # win:1, tie: 0.5, lose: 0
  for index, row in battles_df.iterrows():
    if row['model_1'] not in model_list or row['model_2'] not in model_list:
      continue

    K = 1000
    winner_list = [row['g1_winner'], row['g2_winner']]
    if winner_list.count('model_1') > winner_list.count('model_2'):
      G_1, G_2 = 1 + ((rank_dict[row['model_1']] - rank_dict[row['model_2']]) * 1.0 / K), 0
    elif winner_list.count('model_1') < winner_list.count('model_2'):
      G_1, G_2 = 0, 1 + ((rank_dict[row['model_2']] - rank_dict[row['model_1']]) * 1.0 / K)
    elif winner_list.count('model_1') == winner_list.count('model_2'):
      G_1, G_2 = 0.5, 0.5


    init_G[model_list.index(row['model_1'])] += G_1 * w[model_list.index(row['judge'][0])]
    init_G[model_list.index(row['model_2'])] += G_2 * w[model_list.index(row['judge'][0])]
    error_penalize = 0.05 if winner_list.count('error') > 0 else 0
    init_P[model_list.index(row['judge'][0])] += error_penalize * w[model_list.index(row['judge'][0])]

    log_dict[row['model_1']]['battle'] += 1
    log_dict[row['model_2']]['battle'] += 1
    log_dict[row['judge'][0]]['judge'] += 1

  for i in range(len(model_list)):
    init_G[i] /= log_dict[model_list[i]]['battle']
    init_P[i] /= log_dict[model_list[i]]['judge']
    # init_G[i] -= init_P[i] 
  return init_G

def train(model_list, battles, mode, num_epochs=30, baseline=False):
    battles = battles.sample(frac=1, random_state=seed)
    n = len(model_list)

    if mode == "Uniform":
        init_w = Variable(torch.ones(n)) # Uniform 
    elif mode == "Reversed":
        init_w = torch.arange(0, 1 + 1/(n-1), 1/(n-1)) # Reversed 
    elif mode == "Order":
        init_w = torch.arange(0, 1.0+1.0/n, 1.0/n).flip(0) # Order 
    


    for epoch in range(num_epochs):
        if baseline:
            if mode == "Uniform":
                init_w = Variable(torch.ones(n)) # Uniform 
            elif mode == "Reversed":
                init_w = torch.arange(0, 1 + 1/(n-1), 1/(n-1)) # Reversed 
            elif mode == "Order":
                init_w = torch.arange(0, 1.0+1.0/n, 1.0/n).flip(0) # Order 
            w = Variable(init_w)
            G = cal_score(battles, w, n)
            print("# epoch: " + str(epoch))
            print(w)
            print(G)
            break

        w = Variable(torch.ones(n))
        G = cal_score(battles, w, n)
        print("# epoch: "+str(epoch))
        print(w)
        print(G)
        loss = pearson_correlation(w, G)  
        print(loss)

        if (epoch + 1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

    return w, G


def train_peer(model_list, battles, num_epochs, baseline=False):
    battles = battles.sample(frac=1, random_state=seed)
    n = len(model_list)
    init_w = Variable(torch.randn(n))

    model = nn.Sequential(
        nn.Linear(n, n),
        nn.Sigmoid() 
    )
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    print(model_list)
    for epoch in range(num_epochs):
        w = model(init_w)
        # G = cal_elo(battles, w, n)
        # G = cal_score(battles, w, n)
        G = cal_score_rank(battles, w, n)
        print("# epoch: "+str(epoch))
        print(w)
        print(G)
        loss = pearson_correlation(w, G) 
        print(loss)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)  
        optimizer.step()  
        scheduler.step()    

        if (epoch + 1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

    return w, G


def run_peer_review(epoch, battles):
    logger.info('#' * 10 + ' Peer-' + str(epoch) + '#' * 10)
    w, G = train_peer(model_list, battles, num_epochs=epoch)

    sorted_indices = torch.argsort(G, descending=True).tolist()
    Grade = G.tolist()
    for i in range(len(sorted_indices)):
        ind = sorted_indices[i]
        print('#' + str(i + 1) + ' ' + model_list[ind] + ' | Grade:' + str(Grade[ind]))
        logger.info('#' + str(i + 1) + ' ' + model_list[ind] + ' | Grade:' + str(Grade[ind]))

    print(sorted_indices)
    logger.info(sorted_indices)
    entropy = permutation_entropy(sorted_indices, 3)
    print("Permutation Entropy:", entropy)
    logger.info(f"Permutation Entropy: {entropy}")
    entropy = permutation_entropy_random(sorted_indices, 3)
    print("Random Permutation Entropy:", entropy)
    logger.info(f"Random Permutation Entropy: {entropy}")
    steps = count_bubble_sort_steps(sorted_indices)
    print("Number of Count Inversions", steps)
    logger.info(f"Number of Count Inversions: {steps}")
    steps = longest_increasing_subsequence_length(sorted_indices)
    print("Number of Longest Increasing Subsequence:", steps)
    logger.info(f"Number of Longest Increasing Subsequence:{steps}")

def run_baseline(epoch, battles, mode):
    logger.info('#'*10+' Baseline-'+str(epoch)+'#'*10)
    w, G = train(model_list, battles, mode, num_epochs=1, baseline=True)

    sorted_indices = torch.argsort(G, descending=True).tolist()
    Grade = G.tolist()
    for i in range(len(sorted_indices)):
        ind = sorted_indices[i]
        print('#' + str(i + 1) + ' ' + model_list[ind] + ' | Grade:' + str(Grade[ind]))
        logger.info('#' + str(i + 1) + ' ' + model_list[ind] + ' | Grade:' + str(Grade[ind]))

    print(sorted_indices)
    logger.info(sorted_indices)
    entropy = permutation_entropy(sorted_indices, 3)
    print("Permutation Entropy:", entropy)
    logger.info(f"Permutation Entropy: {entropy}")
    entropy = permutation_entropy_random(sorted_indices, 3)
    print("Random Permutation Entropy:", entropy)
    logger.info(f"Random Permutation Entropy: {entropy}")
    steps = count_bubble_sort_steps(sorted_indices)
    print("Number of Count Inversions", steps)
    logger.info(f"Number of Count Inversions: {steps}")
    steps = longest_increasing_subsequence_length(sorted_indices)
    print("Number of Longest Increasing Subsequence:", steps)
    logger.info(f"Number of Longest Increasing Subsequence:{steps}")


def cal_human_feedback():
    model_list = ['gpt-3.5-turbo', 'vicuna-13b', 'wizardlm-13b', 'vicuna-7b',
                  'koala-13b',
                  'gpt4all-13b-snoozy', 'mpt-7b-chat', 'oasst-pythia-12b', 'alpaca-13b',
                  'stablelm-tuned-alpha-7b', 'dolly-v2-12b', 'llama-13b']

    battles_df = pd.read_json('./one_judge_all/human_feedback.jsonl', lines=True)
    init_G = Variable(torch.zeros(len(model_list)))
    log_dict = {ele: {'battle': 0, 'judge': 0} for ele in model_list}
    # win:1, tie: 0.5, lose: 0
    for index, row in battles_df.iterrows():
        # G_1, G_2
        winner_list = [row['winner']]
        if winner_list.count('model_1') > winner_list.count('model_2'):
            G_1, G_2 = 1, 0
        elif winner_list.count('model_1') < winner_list.count('model_2'):
            G_1, G_2 = 0, 1
        elif winner_list.count('model_1') == winner_list.count('model_2'):
            G_1, G_2 = 0.5, 0.5

        if row['model_1'] not in model_list or row['model_2'] not in model_list:
            continue

        init_G[model_list.index(row['model_1'])] += G_1
        init_G[model_list.index(row['model_2'])] += G_2


        log_dict[row['model_1']]['battle'] += 1
        log_dict[row['model_2']]['battle'] += 1

    for i in range(len(model_list)):
        init_G[i] /= log_dict[model_list[i]]['battle']

    sorted_indices = torch.argsort(init_G, descending=True).tolist()
    Grade = init_G.tolist()
    gt_model_list = []
    for i in range(len(sorted_indices)):
        ind = sorted_indices[i]
        gt_model_list.append(model_list[ind])
        print('#' + str(i + 1) + ' ' + model_list[ind] + ' | Grade:' + str(Grade[ind]))

    print(sorted_indices)
    entropy = permutation_entropy(sorted_indices, 3)
    print("Permutation Entropy:", entropy)
    entropy = permutation_entropy_random(sorted_indices, 3)
    print("Random Permutation Entropy:", entropy)
    steps = count_bubble_sort_steps(sorted_indices)
    print("Number of Count Inversions", steps)
    return gt_model_list


def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_list = ['gpt-3.5-turbo', 'guanaco-33b-merged', 'vicuna-13b-v1.5', 'WizardLM-13B-V1.2', 'vicuna-7b-v1.5','koala-13b', 
                  'gpt4all-13b-snoozy', 'mpt-7b-chat',  'oasst-sft-4-pythia-12b-epoch-3.5', 'alpaca-13b', 'fastchat-t5-3b-v1.0', 'chatglm-6b',
                  'stablelm-tuned-alpha-7b', 'dolly-v2-12b', 'llama-13b']

    parser.add_argument(
        "--baseline",
        type=int,
        default=1,
        help=("determines whether to run the baseline. If set to 0, it runs Peer-review (Ours)")
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="Uniform",
        choices=["Uniform", "Reversed", "Order"],
        help=(
            "Optimization mode. "
            "`Uniform` Setting all LLMs' confidence weights w to 1,. "
            "`Reversed` the model confidence weight w increment from 0 "
            "`Order` decrement confidence weight w from 1 according to the model ranking."
        ),
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=30, 
        help="the number of epochs for conducting Peer-review (Ours)"
    )
    args = parser.parse_args()

    baseline = args.baseline
    print(baseline)
    for seed in [1, 2, 3, 4]:
    # for seed in [4]:
        random.seed(seed)  
        torch.manual_seed(seed)  
        for frac_to_drop in [0.1, 0.4, 0.7, 1]:
            logger_name = 'logger_{}_{}'.format(seed, frac_to_drop)
            if baseline:
                log_file_name = f'./log/frac{frac_to_drop}_{args.mode}_seed{seed}.log'
            else:
                log_file_name = f'./log/frac{frac_to_drop}_Peer_review_seed{seed}.log'
            
            logger = setup_logger(logger_name, log_file_name)

            for seed in [seed]:
                battles = pd.DataFrame()
                for model_str in model_list:
                    print(model_str)
                    df = pd.read_json(f'../llm_judge/data/mt_bench/model_judgment/{model_str}_pair.jsonl', lines=True)
                    df = df.sample(frac=frac_to_drop, random_state=seed)
                    battles = pd.concat([battles, df])

                if baseline:
                    run_baseline(1, battles, args.mode)
                else:
                    run_peer_review(args.epoch, battles)


