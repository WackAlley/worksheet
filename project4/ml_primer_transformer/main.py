import time
from tkinter.font import names

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from encoding_decoding import read_in, alibi_positional_encoding
from transformer import Transformer
from transformer import test
from torch.utils.data import random_split
import torch.nn.functional as F
from icecream import ic


class character_ecode_decode():
    def __init__(self, full_text):
        self.vocab = sorted(set(full_text))  # Liste aller einzigartigen Zeichen im Text
        self.char_to_idx, self.idx_to_char = self.generate_lookup_tables(self.vocab)

    def generate_lookup_tables(self, vocab):
        char_to_idx = {char: idx for idx, char in enumerate(vocab)}
        idx_to_char = {idx: char for idx, char in enumerate(vocab)}
        return char_to_idx, idx_to_char

    def encode(self, text):
        return [self.char_to_idx[char] for char in text]

    def decode(self, indeces):
        return  ''.join([self.idx_to_char[idx.item()] for idx in indeces])


# Dataset erstellen
class CharDataset(Dataset):
    def __init__(self, encoded_text, seq_length):
        self.encoded_text = encoded_text
        self.seq_length = seq_length

    def __len__(self):
        return len(self.encoded_text) - self.seq_length

    def __getitem__(self, idx):
        input_seq = self.encoded_text[idx:idx + self.seq_length]
        target_seq = self.encoded_text[idx + 1:idx + self.seq_length + 1]  # Vorhersage ist das nächste Zeichen
        return torch.tensor(input_seq), torch.tensor(target_seq)


file_path = "goe_full.txt"
full_text = read_in(file_path)
seq_length = 40
encoder = character_ecode_decode(full_text)
encoded_text = encoder.encode(full_text)


# for quick testing: use only a small percentage of all data
dataset = CharDataset(encoded_text[0:int(len(encoded_text)/50)], seq_length) # first 2% of the data

# todo: use full dataset: dataset = CharDataset(encoded_text, seq_length):
#dataset = CharDataset(encoded_text, seq_length) # full dataset

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# todo: use trainnig and testing data, as folloes:
# Split the dataset into 80% training and 20% test data
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = random_split(dataset, [train_size, test_size])


# todo: what is good output? What shell be the length? Only one character?
#todo: write a train and a eval method,
# then train one epoch on the training data, and test with the test data
# loop throug different hyper parameter combinations, and save the models and the results (loss, accuracy)
# one epoch means run through all data once
# example of a simplified training loop is below
# in the file my_net2.py is a example how to train and evalute,
# we should implement a similar thing here
# read_in and plot the losses_df and the accuracy_df (see: plot.py)
# test if expressive attantion, if masking and if alibi positional encoding works and improves the result
# if time left implement beam search:
# output layer has vovabulary sice, each layer stands for a character, normalize it (e.g. with softmax) to get propabilies
# for every output select the x best choises, for each of this choises repete this y times
# multiply properbilities: p = p1 * p2 * p3 ...., choose the one with highest p
#


if __name__ == "__main__":

    # some examples:
    for counter, (input_seq_batch, target_seq_batch) in enumerate(dataloader):
        print("encoded input_seq (batch)", input_seq_batch)
        print("encoded target_seq (batch)", target_seq_batch)
        print("decoded input_seq (batch)", [encoder.decode(input_seq) for input_seq in input_seq_batch])
        print("decoded target_seq (batch)", [encoder.decode(target_seq) for target_seq in target_seq_batch])
        if counter == 5:
            break


    # training
    # Hyperparameter
    embed_size = len(encoder.vocab)
    num_heads = 5
    num_layers = 4
    hidden_size = 512
    epochs = 10
    learning_rate = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Modell, Loss und Optimizer
    model = Transformer(
        embed_size,
        num_layers,
        num_heads,
        device,
        net_expansion_factor=4,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Trainingsloop
    # todo: use positional encoding
    # alibi_bias = alibi_positional_encoding(model.layers[0].attention.n_heads, data.shape[1]).to(model.device)
    alibi_bias = alibi_positional_encoding(num_heads, seq_length).to(device)
    #how long shell output sequence be?
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start = time.time()

        for batch_number, (input_seq, target_seq) in enumerate(dataloader):
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)

            # Vorwärtsdurchlauf
            optimizer.zero_grad()
            input_seq = model.one_hot_encode(input_seq)
            output = model(input_seq)
            target_seq = model.one_hot_encode(target_seq)
            # Berechne den Verlust
            loss = criterion(output.view(-1), target_seq.view(-1))
            running_loss += loss.item()

            # Rückwärtsdurchlauf und Optimierung
            loss.backward()
            optimizer.step()
            if batch_number % 300 == 0:
                print(f"progress: {batch_number/len(dataloader)} % of current_epoch")
        end = time.time()
        # Ausgabe der Verlustwerte pro Epoche
        avg_loss = running_loss / len(dataloader)

        # Test
        test_accuracy, test_loss = test(model, test_data)

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2%}')
        print(f"epoch took {end - start} seconds")


