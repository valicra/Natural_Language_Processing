


import numpy as np
import random
import matplotlib.pyplot as plt

def import_corpus(path_to_file):
    """
    :param path_to_file: Path to corpus file
    :return: List of list of tuples, e.g. [('A', 'DT'), ('Lorillard', 'NNP'), ('spokewoman', 'NN'), ...
    """
    with open(path_to_file) as f:
        return [[tuple(pair.split(' ')) for pair in sent.strip().split("\n")] for sent in f.read().split("\n\n")
                if len(sent) > 0]

def get_key_from_value(my_dict, target_value): #MODIFIED : added this function to retrieve keys from values
    for key, value in my_dict.items():
        if value == target_value:
            return key
    return None


class LinearChainCRF(object):
    def __init__(self, corpus):
        # special label indicating start of sentence
        self.START_LABEL = '<start>'

        # Corpus: [[(Word, Label), (Word, Label), ...],[(Word, Label),(Word,Label),...],...]
        random.shuffle(corpus)
        self.corpus = corpus

        # create train/test split
        num_train_sentences = int(len(corpus) * 0.1) # only use a small fraction of sentences for efficiency
        self.train_sentences = corpus[:num_train_sentences]
        self.dev_sentences = corpus[num_train_sentences:]

        # Create a set of all labels whose count is bigger than Min label count
        self.labels = list({label for s in self.train_sentences for token, label in s})
        self.tokens = list({token for s in self.train_sentences for token, label in s})

        self.labels.append(self.START_LABEL)

        # create the dict feature_indices (Feature to ID)
        self.label_indices = {label: index for index, label in enumerate(self.labels)}
        self.token_indices = {token: index for index, token in enumerate(self.tokens)}

        # initialize theta with 1s (each feature is weighted the same initially)
        self.theta = np.ones((len(self.labels), len(self.labels) + len(self.tokens)))

        # compute empirical feature count initialize with 0s
        self.empirical_count = np.zeros_like(self.theta)
         

        # TODO: Exercise 1e - Precompute the empirical feature count
        #  For each occurrence of a feature in the training data, the corresponding index in self.empirical_count
        #  should be increased by 1
        

        for s in self.train_sentences:
            for i,(token,label) in enumerate(s):
                if i==0:
                    self.empirical_count[self.label_indices[label], self.label_indices[self.START_LABEL]]+=1 # update start|l

                else:
                    self.empirical_count[self.label_indices[label], self.label_indices[s[i-1][1]]]+=1 # update l-1|l
                
                self.empirical_count[self.label_indices[label], len(self.labels) + self.token_indices[token]]+=1 # update l|t






    def psi(self, label, prev_label, token):
        """
        :param label: string with the current label
        :param prev_label: string with the previous label
        :param token: string with the current token
        :return: float for psi
        """
        # TODO: Exercise 1a - Compute psi using the definition of the features from the exercise sheet and self.theta
        
        
        if token not in self.token_indices:
            token = get_key_from_value(self.token_indices, np.argmax(self.empirical_count[self.label_indices[label]])) # get the most frequent token given the label 
        
        ll_index= (self.label_indices[label], self.label_indices[prev_label]) # label| prev_label index
        lt_index= (self.label_indices[label], len(self.label_indices) + self.token_indices[token]) # label|token index
        
        psi = self.empirical_count[ll_index] * self.theta[ll_index] + self.empirical_count[lt_index] * self.theta[lt_index]
    

        return psi


    def forward_variables(self, sentence):
        """
        :param sentence: An input to compute the forward variables/alpha on.
        :return: Data structure containing the matrix of forward variables.
        """
        # TODO: Exercise 1b - Compute the forward variables using your implementation of psi
        #  We recommend to return a 2D array of the shape (len(sentence), len(self.labels))

        alpha = np.zeros((len(sentence),len(self.labels)))
        
        for i, label in enumerate (self.labels):
            alpha[0,i] = self.psi(label, self.START_LABEL, sentence[0][0])
        
        for w, (token, _ ) in enumerate(sentence[1:]): # remember that enumerate strats from 0. here i fix the token. w = word index 
            for j, label in enumerate(self.labels): # fix the label j = label index
                for i, prev_label in enumerate(self.labels): # loop over prev_labels i= looping index 
                    
                    try:
                        alpha[w+1,j]+= self.psi(label, prev_label, token) * alpha[w,i]
                    except RuntimeWarning:
                        alpha[w+1,j] = np.finfo(float).max
                        continue
                
        return alpha

    def backward_variables(self, sentence):
        """
        :param sentence: An input to compute the backward variables/beta on.
        :return: Data structure containing the matrix of forward variables.
        """
        # TODO: Exercise 1b - Compute the backward variables using your implementation of psi
        #  We recommend to return a 2D array of the shape (len(sentence), len(self.labels))

        beta = np.zeros((len(sentence),len(self.labels)))

        for i in range(len(self.labels)):
            beta[len(sentence)-1, i] = 1 

        sentence = list(reversed(sentence))
        for w, (token, _ ) in enumerate(sentence[1:]): # remember that enumerate strats from 0. here i fix the token. w = word index (row)
            for i,prev_label in enumerate(self.labels): # fix the prev_label i = column 
                for j, label in enumerate(self.labels): # loop over labels j = looping index
                    try:
                        beta[len(sentence)-1-(w+1),i]+= self.psi(label, prev_label, token) * beta[len(sentence)-1-w,j]
                    except RuntimeWarning:
                        beta[len(sentence)-1-(w+1),i]=np.finfo(float).max
                        continue
   
        return beta

    def compute_z(self, sentence, alpha_beta):
        """
        :param sentence: A sentence to compute the partition function Z on
        :param alpha_beta: Your alpha or beta variables
        :return: float - Result of the partition function Z
        """
        # TODO: Exercise 1c - Compute the partition function using a datastructure with either
        #  your alpha or beta variables

        alpha=self.forward_variables(sentence)
        z= np.sum(alpha_beta[-1, :])
        return z

    def marginal_probability(self, sentence, t, y_t, y_t_minus_one, alpha, beta, Z):
        """
        Compute the marginal probability of the labels given by y_t and y_t_minus_one given a sentence.
        :param sentence: list of strings representing a sentence.
        :param t: position in sentence for marginal probability, 0-based
        :param y_t: element of the set 'self.labels'; label assigned to the word at position t
        :param y_t_minus_one: element of the set 'self.labels'; label assigned to the word at position t-1
        :param alpha: data structure holding the current alpha variables of the sentence
        :param beta: data structure holding the current beta variables of the sentence
        :param Z: current z value
        :return: float: probability;
        """
        # TODO: Exercise 1d - Compute the marginal probability using the datastructures from forward
        #  and backward, as well as the psi function.

        p = (alpha[t-1,self.label_indices[y_t_minus_one]]*self.psi(y_t, y_t_minus_one, sentence[t][0]) * beta[t,self.label_indices[y_t]] )/ Z
        
        return p

    def expected_feature_count(self, sentence):
        """
        :param sentence: Sentence to compute the expected feature count on
        :return: Data structure holding the expected feature count for each feature
        """
        # TODO: Exercise 1f - Compute the expected feature count for a sentence. We recommend to return a data structure
        #  with the same shape as self.theta. It is given that alpha, beta and Z should first be computed
        alpha = self.forward_variables(sentence)
        beta = self.backward_variables(sentence)
        Z = self.compute_z(sentence, alpha)
        
    

        expected_count=np.zeros_like(self.empirical_count)
        
        for i, (token, label) in enumerate(sentence):
                for prev_label in self.labels:
                    
                    marginal_p= self.marginal_probability(sentence, i,label, prev_label, alpha, beta, Z)
                    
                    expected_count[self.label_indices[label], self.label_indices[prev_label]] = (
                        self.empirical_count[self.label_indices[label], self.label_indices[prev_label]]* marginal_p
                        )
                    expected_count[self.label_indices[label], len(self.label_indices) + self.token_indices[token]] = (
                        self.empirical_count[self.label_indices[label], len(self.label_indices)+self.token_indices[token]] * marginal_p
                    )
        return expected_count

    def train(self, num_iterations, learning_rate=0.01, evaluate_after=20):
        """
        :param num_iterations: Number of training iterations
        :param learning_rate: The learning rate for gradient ascent
        """
        # TODO: Exercise 1g - Implement a training loop over self.training_data that trains for num_iterations.
        #  Every time, each sample has been seen, shuffle the  training data. Print your accuracy on the training
        #  data and on the development data after each iteration. For evaluation you can reuse the function
        #  self.evaluate.
        
        i=1
        j=0
        evaluate=0
        train_accs=[]
        dev_accs=[]
        batches=[]
        
        while i<=num_iterations:
            evaluate+=1
            if evaluate == evaluate_after:
                train_acc, dev_acc = self.evaluate()
                train_accs.append(train_acc)
                dev_accs.append(dev_acc)
                batches.append(i)
                evaluate = 0
                print(f' After {i}/{num_iterations} iterations: ')
                print(f' - Current train_acc is --> {train_acc}')
                print(f' - Current dev_acc is -->  {dev_acc}')
                print('===================')

            if j == len(self.train_sentences):
                random.shuffle(self.train_sentences)
                j=0
            
            train_sentence= self.train_sentences[j]
            new_theta = self.theta + learning_rate * (self.empirical_count - self.expected_feature_count(train_sentence))
            self.theta=new_theta
            i+=1
            j+=1

            # Create subplots with 1 row and 2 columns
        fig, ax = plt.subplots(figsize=(8, 4))

# Plot on the subplot
        ax.plot(batches, train_accs, label='Train')
        ax.plot(batches, dev_accs, label='Dev', color='orange')

        # Set x and y labels
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Accuracy')

        # Add a legend
        ax.legend()

        # Set the title
        ax.set_title('Train vs Dev')

        # Show the plot
        plt.show()





    # Exercise 2 ###################################################################
    def most_likely_label_sequence(self, sentence):
        """
        :param sentence: A sentence we want to get a pos tag sequence for
        :return: A list of predicted labels for the current sentence
        """
        # TODO: Exercise 2 - Compute the most likely label sequence for a given sentence using the viterbi algorithm
        #  We recommend using 2D arrays for gamma and delta
        most_prob_seq=[]
        delta=np.zeros((len(sentence), len(self.labels)))

        for i, label in enumerate (self.labels):
            delta[0,i] = self.psi(label, self.START_LABEL, sentence[0][0])
        
        gamma= get_key_from_value(self.label_indices, np.argmax(delta[0]))
        most_prob_seq.append(gamma)


        for w, (word,_) in enumerate(sentence[1:]):
            for j, label in enumerate(self.labels):
                current_max = 0
                for i, prev_label in enumerate(self.labels):
                    try:
                        candidate_max = self.psi(label, prev_label, word)* delta[w,i] #  delta[w,i] is the previous delta with the previous word 
                    except RuntimeWarning:
                        candidate_max= np.finfo(float).max
                    
                    if candidate_max > current_max:
                        current_max = candidate_max
                delta[w+1, j] = current_max
            gamma =  get_key_from_value(self.label_indices, np.argmax(delta[w+1])) # retrieve the label of the max value for a given word. use w+1 since enuemrate starts from 0 
            most_prob_seq.append(gamma)

                    
        return most_prob_seq

    def evaluate(self, k=30):
        train_sents = self.train_sentences
        dev_sents = self.dev_sentences

        # To reduce the workload of the evaluation method, we select a subset of sentences
        if len(train_sents) > k:
            train_sents = random.choices(self.train_sentences, k=k)
        if len(dev_sents) > k:
            dev_sents = random.choices(self.dev_sentences, k=k)

        train_predictions = [list(zip(self.most_likely_label_sequence(s), [t[1] for t in s])) for s in
                             train_sents]
        dev_predictions = [list(zip(self.most_likely_label_sequence(s), [t[1] for t in s])) for s in
                           dev_sents]

        train_acc = sum([sum([1 if s[0] == s[1] else 0 for s in t]) / len(t) for t in train_predictions]) / len(
            train_predictions)
        dev_acc = sum([sum([1 if s[0] == s[1] else 0 for s in t]) / len(t) for t in dev_predictions]) / len(
            dev_predictions)

        return train_acc, dev_acc

if __name__ == '__main__':
    # Initialize the CRF Object
    corpus = import_corpus('corpus_pos.txt')
    crf = LinearChainCRF(corpus[:120])

    crf.train(500)