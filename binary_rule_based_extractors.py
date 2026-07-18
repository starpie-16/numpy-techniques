import numpy as np
import re


def binary_rule_based_extractors(spam, banned_words):

  #tokenization & indexing

  tokenized_spam = [sentence.split() for sentence in spam] # 2d 
  all_words = [word for sentence in tokenized_spam for word in sentence]

  vocab = np.unique(all_words)
  word_to_idx = {w : idx for idx, w in enumerate(vocab)} #dict

  indexed_sentences = [[word_to_idx[word] for word in sentence] for sentence in tokenized_spam] # 2d
  banned_w_indices = np.array([word_to_idx[w] for w in banned_words if w in word_to_idx]) #1d 

  #padding

  max_len = max(len(s) for s in tokenized_spam)
  padded_array = np.full((len(spam), max_len), fill_value = -1) # (num_of_sent, max_len)

  for i, s_indices in enumerate(indexed_sentences):
    padded_array[i, :len(s_indices)] = s_indices
  
  #return padded_array, banned_indices

  #binary_rule_based_extractors

  comparision = padded_array[:, :, None] == banned_w_indices[None, None, :]

  final_matrix = np.any(comparision, axis = 1).astype(int)

  return final_matrix
