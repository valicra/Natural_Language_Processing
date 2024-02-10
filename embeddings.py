#!/usr/bin/env python
import gensim.downloader as api
from gensim.models.keyedvectors import KeyedVectors
from scipy.stats import spearmanr
from scipy.spatial import distance

def load_ws353_dataset(path):
    with open(path, "r") as file:
        # skip the first line
        file = iter(file)
        next(file)
        word_pairs = []
        gold_labels = []
        for line in file:
            w1, w2, human = line.split("\t") 
            word_pairs.append((w1, w2))
            gold_labels.append(float(human))
    return word_pairs, gold_labels


def load_embeddings(path):
    # TODO: Exercise (1.a) 
    model= api.load(path)
    return model 

def compute_cosine_similarities(word_pairs, word_vectors):
    # TODO: Exercise (1.b1) 
    distances=[]
    
    for pair in word_pairs:
        v0 = word_vectors[pair[0]]
        v1 = word_vectors[pair[1]]
        dist = distance.cosine(v0,v1)
        distances.append(dist)
        
    
    return distances
    
def compute_euclidean_distance(word_pairs, word_vectors):
    # TODO: Exercise (1.b2)
    distances=[]
    
    for pair in word_pairs:
        v0 = word_vectors[pair[0]]
        v1 = word_vectors[pair[1]]
        dist = distance.euclidean(v0,v1)
        distances.append(dist)
        
    return distances
    


def compute_spearman(gold_labels, prediction):
    # TODO: Exercise (1.c1)
    return spearmanr(gold_labels, prediction)[0]
    

def compute_bias(biased_pairs, word_vectors):
    word1, word2, word3 = biased_pairs
    # TODO: Exercise (1.d)
    result = word_vectors.most_similar(positive=[word1, word2], negative=[word3])
    return result[:5]


# ------------------------------------------------------------------------------

if __name__ == "__main__":
    word_pairs, gold_labels = load_ws353_dataset("datasets/wordsim353.txt")
    embeddings = load_embeddings("word2vec-google-news-300")
    predictions_cosine = compute_cosine_similarities(word_pairs, embeddings)
    predictions_euclidean = compute_euclidean_distance(word_pairs, embeddings)
    print("Spearman's r cosine similarity :", round(compute_spearman(gold_labels, predictions_cosine), 4))
    print("Spearman's r euclidean distance:", round(compute_spearman(gold_labels, predictions_euclidean), 4))
    
    print("# TODO: Exercise (1.c2) Please add your comments here")
    #Both coefficients are negative, suggesting a negative trend between values assigned from human annotaters and the pnes from the model. 
    #This probably come from the techniques used by the annotators to define similarity. 
    #This is clear with the example of "midday" and "noon" which are given a high similarty (9.29/10) by the humans, due to their meaning, even though they are quite different literally. 
    #Conversely, the cosine similarity assigns just 0.45/1 to this pair. 
    #There are also othere examples showing high similarity assigned by cosine and lower similarity assigned by humans. 
    
    print(compute_bias(["computer_programmer", "woman", "man"], embeddings))
    print(compute_bias(["computer_programmer", "man", "woman"], embeddings))
    
