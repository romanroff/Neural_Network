# ------------------------ Подстановка пропущенных слов ------------------------ #
import sys,random,math
from collections import Counter
import numpy as np


def similar(target='beautiful'):
  target_index = word2index[target]

  scores = Counter()
  for word,index in word2index.items():
    raw_difference = weight01[index] - (weight01[target_index])
    squared_difference = raw_difference * raw_difference
    scores[word] = -math.sqrt(sum(squared_difference))
  return scores.most_common(10)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


np.random.seed(1)
random.seed(1)
f = open('reviews.txt')
raw_reviews = f.readlines()
f.close()

tokens = list(map(lambda x:(x.split(" ")),raw_reviews))
wordcnt = Counter()
for sent in tokens:
    for word in sent:
        wordcnt[word] -= 1
vocab = list(set(map(lambda x:x[0],wordcnt.most_common())))

word2index = {}
for i,word in enumerate(vocab):
    word2index[word]=i

concatenated = list()
input_dataset = list()
for sent in tokens:
    sent_indices = list()
    for word in sent:
        try:
            sent_indices.append(word2index[word])
            concatenated.append(word2index[word])
        except:
            ""
    input_dataset.append(sent_indices)
concatenated = np.array(concatenated)
random.shuffle(input_dataset)

alpha, iterations = (0.05, 2)
hidden_size,window,negative = (50,2,5)

weight01 = (np.random.rand(len(vocab),hidden_size) - 0.5) * 0.2
weight12 = np.random.rand(len(vocab),hidden_size)*0

layer_2_target = np.zeros(negative+1)
layer_2_target[0] = 1
#
#
for rev_i, review in enumerate(input_dataset * iterations):
    for target_i in range(len(review)):

        target_samples = [review[target_i]] + list(concatenated[(np.random.rand(negative) * len(concatenated)).astype('int').tolist()])

        left_context = review[max(0, target_i - window):target_i]
        right_context = review[target_i + 1:min(len(review), target_i + window)]

        layer_1 = np.mean(weight01[left_context+right_context], axis=0)
        layer_2 = sigmoid(layer_1.dot(weight12[target_samples].T))
        layer_2_delta = layer_2 - layer_2_target
        layer_1_delta = layer_2_delta.dot(weight12[target_samples])

        weight01[left_context+right_context] -= layer_1_delta * alpha
        weight12[target_samples] -= np.outer(layer_2_delta, layer_1) * alpha

    if rev_i % 250 == 0:
        sys.stdout.write(f'\rProgress: {rev_i / float(len(input_dataset) * iterations)} {similar("terrible")}')
    sys.stdout.write(f'\rProgress: {rev_i / float(len(input_dataset) * iterations)} {similar("terrible")}')

print(similar('terrible'))