import torch
import re
import math

# read and tokenize text
# from https://itp.uni-frankfurt.de/~gros/Vorlesungen/ML/2024_ML02_Generative_Architectures.html#(25)
file_in = open('goethe.txt', 'r', encoding='utf-8')
trainingString = file_in.read()
file_in.close()
trainingString = re.sub(r'[^\w\s]', '', trainingString) # interpunction removal via regular expression
trainingString = trainingString.lower() # lowercase

trainingText = list(trainingString)
vocabulary = list(set(trainingText))
dim = len(vocabulary)

# one hot embedding
letterEmbedding = {letter: torch.zeros(dim) for letter in vocabulary}

count = 0
for letter in vocabulary:
    letterEmbedding[letter][count] = 1
    count += 1


# ALiBi positional encoding
# https://itp.uni-frankfurt.de/~gros/Vorlesungen/ML/2024_ML02_Generative_Architectures.html#(11)

nC = 6
nHead = 2

rel_dist = torch.arange(0, nC).view(1, 1, nC) - torch.arange(0, nC).view(1, nC, 1)    
slopes = torch.tensor([1.0/(2.0**(h*1.0/nHead)) for h in range(nHead)])
biases = -slopes.view(nHead, 1, 1) * rel_dist.abs() 
ALiBi_tensor = biases.exp()

# Nächster Schritt ist es den ALiBi Tensor im linearen Attention zu addieren.
# Q_j * K_i -> Q_j * K_i + m(i-j), j>= i (causal), wobei m gleich slopes ist. Jedoch wird nicht der slope, sondern der bias dazuaddiert, was mir noch unklar ist

# zufällige Werte, die durch Lernen ersetzt werden
K = torch.rand(nC, nHead)  # Key
Q = torch.rand(nC, nHead)  # Query

linear_attention = torch.matmul(Q,K.T) + ALiBi_tensor


