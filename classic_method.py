from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from string import punctuation
from heapq import nlargest


def compute_freq(word_sent):
    freq = defaultdict(int)
    stop_words = set(stopwords.words('english') + list(punctuation))
    min_cut = 0.1
    max_cut = 0.9
    
    for s in word_sent:
      for word in s:
        if word not in stop_words:
          freq[word] += 1
#     frequencies normalization and fitering
    m = float(max(freq.values()))
    for w in list(freq):
      freq[w] = freq[w]/m
      if freq[w] >= max_cut or freq[w] <= min_cut:
        del freq[w]
    return freq

def rank(ranking, n):
    return nlargest(n, ranking, key=ranking.get)


def summarize(text, n):
    sents = sent_tokenize(text)
#    assert n <= len(sents)
    word_sent = [word_tokenize(s.lower()) for s in sents]
    freq = compute_freq(word_sent)
    ranking = defaultdict(int)
    for i,sent in enumerate(word_sent):
      for w in sent:
        if w in freq:
          ranking[i] += freq[w]
    sents_idx = rank(ranking, n)    
    return [sents[j] for j in sents_idx]


with open('text.txt', 'r') as file:
    data = file.read()
    for s in summarize(data, 2):
        print('*', s)
        
        
        
        
        
        
        