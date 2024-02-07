import numpy as np
import math
import random
from collections import Counter
import copy


def permutation_entropy(x, m, t=1):
    x = np.array(x)
    if len(x) < m:
        raise ValueError("Error")
    if t > m:
        t = m
    X = []
    if t == 1:
        length = int(len(x) - m+1)
    else:
        length = int((len(x) - m + 1) / t) + 1

    for i in range(length):
        X.append(x[i*t:i*t+m])
    loop = 1
    index = []
    for i in X:
        index.append(str(np.argsort(i)))

    entropy = [0]*loop
    for temp in range(loop):
        count = Counter(index)
        for i in count.keys():
            entropy[temp] += -(count[i]/len(index))*math.log(count[i]/len(index), math.e)
    return entropy

def calculate_combinations(n, k):
    combinations = math.factorial(n) / (math.factorial(k) * math.factorial(n - k))
    return int(combinations)

def permutation_entropy_random(x, m, t=1):

    x = np.array(x)
    if len(x) < m:
        raise ValueError("Error")

    if t > m:
        t = m

    X = []
    if t == 1:
        length = int(len(x) - m+1)
    else:
        length = int((len(x) - m + 1) / t) + 1

    T = calculate_combinations(len(x), m)
    for _ in range(T):
        sampled_indices = random.sample(range(len(x)), m)
        sorted_sampled_indices = sorted(sampled_indices)
        sorted_sampled_elements = [x[i] for i in sorted_sampled_indices]

        X.append(sorted_sampled_elements)

    loop = 1
    index = []
    for i in X:
        index.append(str(np.argsort(i)))

    entropy = [0]*loop
    for temp in range(loop):
        count = Counter(index)
        for i in count.keys():
            entropy[temp] += -(count[i]/len(index))*math.log(count[i]/len(index), math.e)
    return entropy


def count_bubble_sort_steps(arr):
    arr = copy.deepcopy(arr)
    n = len(arr)
    steps = 0

    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                steps += 1

    return steps

def longest_increasing_subsequence_length(nums):
    n = len(nums)
    dp = [1] * n

    for i in range(1, n):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)

if __name__ == "__main__":
    arr = [2, 0, 1, 3, 4, 11, 5, 7, 8, 6, 10, 12, 9, 13, 14]
    print(longest_increasing_subsequence_length(arr))