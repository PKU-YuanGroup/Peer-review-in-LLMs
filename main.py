import os
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from scipy.stats import spearmanr
import numpy as np
from itertools import permutations
from con_optimization.utils import *
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler

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
    # score G_1, G_2
    winner_list = [row['g1_winner'], row['g2_winner']]
    if winner_list.count('model_1') > winner_list.count('model_2'):
      G_1, G_2 = 1, 0
    elif winner_list.count('model_1') < winner_list.count('model_2'):
      G_1, G_2 = 0, 1
    elif winner_list.count('model_1') == winner_list.count('model_2'):
      G_1, G_2 = 0.5, 0.5

    if row['model_1'] not in model_list or row['model_2'] not in model_list:
      continue
    ## update scores
    init_G[model_list.index(row['model_1'])] += G_1 * w[model_list.index(row['judge'][0])]
    init_G[model_list.index(row['model_2'])] += G_2 * w[model_list.index(row['judge'][0])]
    error_penalize = 0.05 if winner_list.count('error') > 0 else 0
    init_P[model_list.index(row['judge'][0])] += error_penalize * w[model_list.index(row['judge'][0])]

    ## update log
    log_dict[row['model_1']]['battle'] += 1
    log_dict[row['model_2']]['battle'] += 1
    log_dict[row['judge'][0]]['judge'] += 1

  for i in range(len(model_list)):
    init_G[i] /= log_dict[model_list[i]]['battle']
    init_P[i] /= log_dict[model_list[i]]['judge']

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

    K = 200
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

def train(model_list, battles, num_epochs=30, baseline=False):
    battles = battles.sample(frac=1, random_state=4)
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
        if baseline:
            w = Variable(torch.ones(n))
            G = cal_score(battles, w, n)
            print("# epoch: " + str(epoch))
            print(w)
            print(G)
            break
        # G = cal_elo(battles, w, n)
        # G = cal_score(battles, w, n)
        G = cal_score_rank(battles, w, n)
        print("# epoch: "+str(epoch))
        print(w)
        print(G)
        loss = pearson_correlation(w, G)  # loss
        print(loss)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)  
        optimizer.step()  
        scheduler.step()    

        if (epoch + 1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

    return w, G


def run_peer_review():
    w, G = train(model_list, battles, num_epochs=30)

    # sort
    sorted_indices = torch.argsort(G, descending=True).tolist()
    Grade = G.tolist()
    for i in range(len(sorted_indices)):
        ind = sorted_indices[i]
        print('#' + str(i + 1) + ' ' + model_list[ind] + ' | Grade:' + str(Grade[ind]))

    # Calculate Permutation Entropy
    print(sorted_indices)
    entropy = permutation_entropy_random(sorted_indices, 3)
    print("Permutation Entropy:", entropy)
    steps = count_bubble_sort_steps(sorted_indices)
    print("Number of Bubble Sort Iterations:", steps)
    steps = longest_increasing_subsequence_length(sorted_indices)
    print("Number of Longest Increasing Subsequence:", steps)

def run_baseline():
    w, G = train(model_list, battles, num_epochs=1, baseline=True)

    # sort
    sorted_indices = torch.argsort(G, descending=True).tolist()
    Grade = G.tolist()
    for i in range(len(sorted_indices)):
        ind = sorted_indices[i]
        print('#' + str(i + 1) + ' ' + model_list[ind] + ' | Grade:' + str(Grade[ind]))

    # Calculate Permutation Entropy
    print(sorted_indices)
    entropy = permutation_entropy_random(sorted_indices, 3)
    print("Permutation Entropy:", entropy)
    steps = count_bubble_sort_steps(sorted_indices)
    print("Number of Bubble Sort Iterations:", steps)
    steps = longest_increasing_subsequence_length(sorted_indices)
    print("Number of Longest Increasing Subsequence:", steps)


if __name__ == "__main__":

    model_list = ['gpt-3.5-turbo', 'guanaco-33b-merged', 'vicuna-13b-v1.5', 'WizardLM-13B-V1.2', 'vicuna-7b-v1.5','koala-13b', 
                  'gpt4all-13b-snoozy', 'mpt-7b-chat',  'oasst-sft-4-pythia-12b-epoch-3.5', 'alpaca-13b', 'fastchat-t5-3b-v1.0', 'chatglm-6b',
                  'stablelm-tuned-alpha-7b', 'dolly-v2-12b', 'llama-13b']
    battles = pd.DataFrame()
    for model_str in model_list:
        print(model_str)
        df = pd.read_json('./llm_judge/data/mt_bench/model_judgment/' + model_str + '_pair.jsonl', lines=True).sort_values(ascending=True,
                                                                                                        by=["tstamp"])
        battles = pd.concat([battles, df])

    run_baseline()
    run_peer_review()