import numpy as np
import pandas as pd
import re
import nltk
nltk.download('punkt') # one time execution
nltk.download('stopwords') # one time execution
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import networkx as nx
from nltk.cluster.util import cosine_distance

###################### read dataset split to sentences #################################

def read_clean_dataset(dataset,column_name):

    df = pd.read_csv(dataset) # dataset name #
    sentences = []              
    for s in df[column_name]: #column name# from the dataset that has the text to summ
        sentences.append(sent_tokenize(s))
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")  # remove punctuations, numbers and special characters 

    return clean_sentences
        
def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)

############################ build_similarity_matrix  ####################################

def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


######################### generate summary ##############################
def generate_summary(dataset,column_name, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []

    # Read text  split it
    sentences =  read_clean_dataset(dataset,column_name)

    # Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    print("Indexes of top ranked_sentence order are ", ranked_sentence)    

    for i in range(top_n):
      summarize_text.append(" ".join(ranked_sentence[i][1]))

    # Offcourse, output the summarize texr
    print("Summarize Text: \n", ". ".join(summarize_text))



generate_summary("Dataset","column_name", 2)  # replace  "dataset" and  "column_name" with the dataset and column names












