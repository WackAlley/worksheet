import re
import string
import torch
import torch.nn.functional as F
import numpy
from icecream import ic

encoding = "Latin-1"

def get_char_vocabulary(string):
    unique_tokens = sorted(set(string))
    return ''.join(char for char in unique_tokens)

def read_in(file_path):
    with open(file_path,'r', encoding="utf-8") as file:\
        text = file.read()
    #sustitute non standard characters
    cleaned_text = text.replace(r'’', r"'").replace(r'”', r'"')
    ignore_chars = string.punctuation + '\\\«»\n' #möglicherweise erlauben: !"':;?
    non_german_chars = "ÇÈàâèéêëòóùû"
    #remove words with non german characters and all ignore chars
    cleaned_text = re.sub(rf'(\b\w*[{non_german_chars}]\w*\b|[{ignore_chars}])', ' ', cleaned_text) # remove words with: ÇÈàâèéêëòóùû
    cleaned_text = re.sub(r' +', ' ', cleaned_text) # one or more ' ' replaced by one
    return cleaned_text

class character_one_hot_embedding():
    def __init__(self, vocabulary_string):
        vocabulary_tensor = torch.tensor(bytearray(vocabulary_string, encoding), dtype=torch.int)
        is_char_in_vocab = torch.full((vocabulary_tensor.max().item()+1,), False, dtype=torch.bool)
        is_char_in_vocab[vocabulary_tensor] = True
        self.vocabulary_size = torch.count_nonzero(is_char_in_vocab).item() # False = 0, True = 1, count all Trues
        self.encoding_lookup = torch.cumsum(is_char_in_vocab == True, dim=0) - 1
        self.decoding_lookup = torch.arange(is_char_in_vocab.size(dim=0), dtype=torch.int)[is_char_in_vocab] # use is_char_in_vocab as mask

    def encode(self, input_text):
        encoded_indices = self.encoding_lookup[torch.tensor(bytearray(input_text, encoding), dtype=torch.int)]
        one_hot_vectors = F.one_hot(encoded_indices, num_classes=self.vocabulary_size).float()
        return one_hot_vectors

    def decode(self, one_hot_vectors):
        decoded_indices = self.decoding_lookup[torch.argmax(one_hot_vectors, dim=1)]
        reconstructed_text = decoded_indices.to(torch.uint8).numpy().tobytes().decode(encoding)
        return reconstructed_text


def alibi_positional_encoding(n_heads,sequence_length):
    rel_dist = torch.arange(0, sequence_length).view(1, 1, sequence_length) - torch.arange(0, sequence_length).view(1, sequence_length, 1)
    #slopes = torch.tensor([1.0 / (2.0 ** (h * 1.0 / n_heads)) for h in range(n_heads)])
    slopes = 1.0 / (2.0 ** (torch.arange(n_heads, dtype=torch.float32) / n_heads))
    biases = -slopes.view(n_heads, 1, 1) * rel_dist.abs()
    ALiBi_tensor = biases.exp()
    return ALiBi_tensor


if __name__ == "__main__":
    file_path = "goe_full.txt"
    full_text = read_in(file_path)
    vocab = get_char_vocabulary(full_text)
    encoder = character_one_hot_embedding(vocab)
    #torch.set_printoptions(threshold=10000)
    print(encoder.encode(vocab))
    text = "Wir müssen Faust finden und ihm sagen dass wir ihn ab sofort Hand nennen"
    text = vocab
    vectors = encoder.encode(text)
    print(vectors.size())
    #print(vectors)
    #print("vectors", vectors := encoder.encode(text))
    print(encoder.decode(vectors))
    print(encoder.vocabulary_size)
    print(vocab)
