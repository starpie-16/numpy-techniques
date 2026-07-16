import numpy as np

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    
    
    words = [w for s in documents for w in s.split()]
    vocab = np.unique(words)
    
    

    #indexing
    word_to_idx = {w: idx for idx, w in enumerate(vocab)}
    indexed_docs = [[word_to_idx[w] for w in s.split()] for s in documents]
  

    #padding 
    max_len = max(len(s) for s in indexed_docs)
    padded_sent = np.full((len(indexed_docs), max_len), fill_value = -1)

    for i, sent in enumerate(indexed_docs):
        padded_sent[i, :len(sent)] = sent



    N = len(documents)
    V = len(vocab)

    

    # tf
    vocab_size = len(vocab)
    vocab_indices = np.arange(V)

    comparision = padded_sent[:, :, np.newaxis] == vocab_indices[np.newaxis, np.newaxis, :]
    freq = np.sum(comparision, axis = 1)

    row_sum = np.sum(freq, axis = 1)[:, None]
    tf = freq / row_sum

    # idf
    df_bool = freq > 0
    df = np.sum(df_bool, axis = 0)
    idf = np.log(N / (df))

    tfidf_matrix = tf * idf

    return tfidf_matrix, vocab
