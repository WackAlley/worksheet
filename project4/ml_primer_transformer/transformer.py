
from encoding_decoding import character_one_hot_embedding, alibi_positional_encoding
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from icecream import ic
import numpy


class SelfAttention(nn.Module):
    """
    Implements multi-head self-attention mechanism similar to the one described in
    the paper "Attention Is All You Need" by Vaswani et al.

    This module projects input embeddings into queries, keys, and values,
    calculates attention scores, and combines attended values for further processing.

    During initialization there is the option to choose between two types of attention mechanisms:
    1. dot-product attention: (as described in the paper "Attention Is All You Need"):
       This is the standard dot-product attention mechanism, where the attention scores are computed based on the dot product between queries and keys, followed by softmax.
    2. expressive attention: (as proposed by Claudius Gros in the paper "Reorganizing Attention-Space Geometry with Expressive Attention"):
       The attention score in this approach is a function of the squared dot product between queries and keys. In leading order, expressive attention is
       quartic in the token activities, making it fundamentally different from standard linear attention, which is bilinear in token activities.

    Attributes:
    -----------
    embedding_dim : int
        The dimension (size) of the embedding vector space.
    n_heads : int
        The number of attention heads; the embedding is divided (lineary projected) into this many parts.
    head_size : int
        The dimension of each attention head (embedding_dim // n_heads).
    attention_type : str (default='dot_product')
        Type of attention mechanism to use. Options: 'dot_product', 'expressive'.
        The 'expressive' option refers to the approach presented in
        "Reorganizing Attention-Space Geometry with Expressive Attention" by Claudius Gros.

    Methods:
    --------
    forward(input, alibi_bias=None, mask=None):
        Applies self-attention to the input tensor with optional bias and mask.
    """
    def __init__(self, embedding_dim, n_heads, attention_type = 'dot_product'):
        """
        embedding_dim: int
            The dimension of the embedding vector space.
        n_heads: int
            The number of attention heads; the embedding vector is divided into this many parts.
       attention_type: str, optional (default='dot_product')
            Type of attention mechanism to use. Options: 'dot_product', 'expressive'
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.head_size = embedding_dim // n_heads
        assert (
                self.head_size * n_heads == embedding_dim
        ), f"Embedding size needs to be divisible by n_heads. Embedding size: {embedding_dim}, n_heads: {n_heads}"

        assert (attention_type in ['dot_product', 'expressive']
                ), f"Invalid attention type. Choose 'dot_product' or 'expressive'. Got: {attention_type}"

        self.attention_type = attention_type

        # three learnable affine (linear) transformations to transform input into values, keys and querrys
        self.value_transform = nn.Linear(embedding_dim, embedding_dim) # , bias=False) # set bias=False to do a pure matrix multipliation
        self.key_transform = nn.Linear(embedding_dim, embedding_dim) # , bias=False)
        self.query_transform = nn.Linear(embedding_dim, embedding_dim) # , bias=False)

        self.fc_out = nn.Linear(embedding_dim, embedding_dim)  # fc stands for fully connected
        # heads = number of heads, embeding is split in #head parts, head_dim = head_size; head_dim * heads = embede_size

    def forward(self, input ,alibi_bias = None, mask = None):
        """
        input: torch.Tensor
            The input tensor of shape (batch_size, seq_length, embedding_dim).
        alibi_bias : torch.Tensor, optional
            Optional bias tensor for attention scores, used for relative positioning, used for ALiBi positional encoding.
            Shape can be (n_heads, query_len, key_len) or (batch_size, n_heads, query_len, key_len).
        mask: torch.Tensor, optional
            Optional mask tensor to ignore certain positions in the sequence,
            use case: causality, ensuring each position can only attend to previous positions.

        return: torch.Tensor
            The output tensor of shape (batch_size, seq_length, embedding_dim),
            containing the attended values after applying self-attention.
        """

        values = self.value_transform(input)  # (batch_dim, value_len, embedding_dim)
        keys = self.key_transform(input)  # (batch_dim, key_len, embedding_dim)
        queries = self.query_transform(input)  # (batch_dim, query_len, embedding_dim)

        # Get number of examples (batches)
        batch_dim = queries.shape[0]  # how many examples we send in at the same time (in one batch)

        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split the embedding into self.n_heads different pieces
        values = values.reshape(batch_dim, value_len, self.n_heads, self.head_size)
        keys = keys.reshape(batch_dim, key_len, self.n_heads, self.head_size)
        queries = queries.reshape(batch_dim, query_len, self.n_heads, self.head_size)


        # Here Einsum does matrix mult. query*keys for each training example
        # Einsum is a handy way to do Transpose and Matrix multiplications in one step
        # multiplies dimensions with same indecs summs over indecs which do not appear in the result
        attention_scores = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # d is head size (head dimension)
        # queries shape: (batch_dim, query_len, n_heads, heads_size),
        # keys shape: (batch_dim, key_len, n_heads, heads_size)
        # attention_scores shape: (batch_dim, n_heads, query_len, key_len)

        # example: if query is  source target sentence and key source sentence, then attention_scores would tell us
        # for each word in target sentence how much attention should we pay to each word in source sentence
        if alibi_bias is not None:
            if alibi_bias.dim() == 3:
                # Shape: (n_heads, query_len, key_len) → Add batch dimension
                assert alibi_bias.shape == (n_heads, query_len, key_len
                    ), f"Invalid ALiBi bias shape. Expected: ({n_heads}, {query_len}, {key_len}), but got: {alibi_bias.shape}"
                # in this case a broadcast will happen, because torch.add supports broadcast
                # to all bach dimensions this mask will be applied
            elif alibi_bias.dim() == 4:
                # Shape: (batch_size, n_heads, query_len, key_len)
                assert alibi_bias.shape == (batch_dim, n_heads, query_len, key_len
                    ), f"Invalid ALiBi bias shape with batch dimension. Expected: {batch_dim}, {n_heads}, {query_len}, {key_len}), but got: {alibi_bias.shape}"
            else:
                assert f"Invalid number of dimensions for ALiBi bias: expected 3 or 4, but got {alibi_bias.dim()}"

            attention_scores = attention_scores + alibi_bias

        # Mask padded indices so their weights become 0
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float("-1e20"))  # set all 0 entries to verry small value to avoid over-/ underflow later on


        if self.attention_type == 'dot_product':
            # Normalize attention_scores. Also divide by scaling factor for better stability
            # todo: Is additional normalisation needed?
            attention = torch.softmax(attention_scores / (self.embedding_dim ** (1 / 2)),
                                      dim=3)  # implements formular 1 of attention is all you need paper
            # attention shape: (batch_dim, n_heads, query_len, key_len)
        elif self.attention_type == 'expressive':
            # Apply the expressive attention formula
            attention = (attention_scores ** 2) / (1 + attention_scores ** 2)
            # todo: Is normalisation Faktor N_m needed and correct?
            # Normalize by N_m (sum over the sequence dimension)
            N_m = attention.sum(dim=-1, keepdim=True)  # Sum over keys dimension
            attention = attention / N_m

        # key length and value lenth are l, multiply along this l-dimension
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            batch_dim, query_len, self.n_heads * self.head_size
        )  # the reshape concatinates all heads
        # attention shape: (batch_dim, n_heads, query_len, key_len)
        # values shape: (batch_dim, value_len, n_heads, heads_size)
        # out after matrix multiply: (batch_dim, query_len, n_heads, head_size), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape is
        # (batch_dim, query_len, embedding_dim)
        return out


class TransformerLayer(nn.Module):
    """
    Implements a single layer of the Transformer model, which mainly consists of  a Multi-head self-attention
    and a Feed-forward network component
    Each component has a residual connection followed by LayerNorm for better gradient flow and stability during training.

    Attributes:
    -----------
    attention: SelfAttention
        The multi-head self-attention mechanism used in the transformer layer.
    norm1: nn.LayerNorm
        Layer normalization applied directly before the attention mechanism.
    norm2: nn.LayerNorm
        Layer normalization applied directly before the feed-forward network.
    embedding_dim: int
        The size of the input embeddings.
    feed_forward: nn.Sequential
        The feed-forward network consisting of two linear layers with a ReLU activation in between.

    Methods:
    --------
    forward(input, alibi_bias=None, mask=None):
        Applies the transformer layer to the input tensor, including attention, feed-forward network, and normalization.
    """

    def __init__(self, embedding_dim, n_heads, net_expansion_factor, attention_type='dot_product'):
        """
        embedding_dim : int
            The size of the embedding vector space (i.e., the dimension of each token's embedding).
        n_heads : int
            The number of attention heads used in the multi-head self-attention mechanism.
        net_expansion_factor : int
            The factor by which the number of nodes is expanded in the feed-forward network.
        attention_type : str, optional (default='dot_product')
            Type of attention mechanism to use. Options: 'dot_product' for standard attention or 'expressive' for expressive attention.
        """
        super().__init__()
        self.attention = SelfAttention(embedding_dim, n_heads, attention_type = attention_type)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.embedding_dim = embedding_dim

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, net_expansion_factor * embedding_dim),  # expand
            nn.ReLU(),
            nn.Linear(net_expansion_factor * embedding_dim, embedding_dim),  # squish
        )

    def forward(self, input, alibi_bias = None, mask = None):
        """
        input: torch.Tensor
            The input tensor of shape (batch_size, seq_length, embedding_dim), where `batch_size` is the number of sequences in a batch,
            `seq_length` is the length of the sequence, and `embedding_dim` is the dimensionality of each token embedding.
        alibi_bias: torch.Tensor, optional
            Optional bias tensor for attention scores, used for relative positioning.
            Shape can be (n_heads, query_len, key_len) or (batch_size, n_heads, query_len, key_len).
        mask: torch.Tensor, optional
            Optional mask tensor to ignore certain positions in the sequence, used for causality or padding.
            Shape should be (batch_size, 1, 1, key_len).

        return: torch.Tensor
            The output tensor of shape (batch_size, seq_length, embedding_dim), containing the processed representation after applying
            self-attention, feed-forward network, and residual connections.
        """

        assert (input.dim() == 3),f"Expected input of shape (batch_size, len, embedding_dim) but got shape {input.shape}"
        assert (input.shape[2] == self.embedding_dim), "Transformer input needs to be of size embedding_dim (for each batch)."

        normed_input = self.norm1(input)
        #  affine(linear) transformations to transform input into values, keys and querrys are implemented in attention mechanism
        #  In that way only one tensor (input) needs to pe passed, instead of three (calue, key, querry)
        attention = self.attention(normed_input, alibi_bias = alibi_bias, mask = mask)
        x = attention + input  # skipp connection for attention

        normed_x = self.norm2(x)
        forward = self.feed_forward(normed_x)
        out = forward + x  # skipp connection for feed forward
        return out

class Transformer(nn.Module):
    """
    Transformer model consisting of multiple TransformerLayers.This model processes sequences of tokens and produces
     an output representation for each token in the sequence. The input sequence is passed through each layer in turn.

    The Transformer architecture relies on the self-attention mechanism, which allows the model to weigh the importance
    of different tokens in a sequence in relation to each other.

    Attributes:
    -----------
    embedding_dim : int
        The size of the input/output embedding space (i.e., the dimensionality of each token's embedding).
    device : str
        The device on which the model should run, either "cpu" or "cuda".
    layers : nn.ModuleList
        A list of TransformerLayer instances, each of which applies multi-head self-attention followed by a feed-forward network.
    norm : nn.LayerNorm
        A LayerNorm applied to the output before all the final linear layer.
    linear_fc_out : nn.Linear
        A final fully connected linear layer that projects the output to the original embedding dimension.

    Methods:
    --------
    forward(input, mask=None):
        Applies the Transformer model to the input tensor. It processes the input through multiple transformer layers,
        applies normalization, and generates the output representation for each token in the input sequence.

    one_hot_encode(input):
        Converts the input tensor into a one-hot encoded representation, where each token is represented as a one-hot vector.

    one_hot_decode(one_hot_vectors):
        Decodes the one-hot encoded vectors back into token indices.
    """
    def __init__(
        self,
        embedding_dim,
        num_layers,
        n_heads,
        device = "cpu",
        net_expansion_factor = 4,
        attention_type='dot_product'
    ):
        """
        embedding_dim : int
            The dimensionality of the token embeddings.
        num_layers : int
            The number of TransformerLayer instances to stack.
        n_heads : int
            The number of attention heads for the multi-head self-attention mechanism.
        device : str, optional (default='cpu')
            The device to run the model on. Should be either 'cpu' or 'cuda'.
        net_expansion_factor : int, optional (default=4)
            The expansion factor for the feed-forward network inside each TransformerLayer.
        attention_type : str, optional (default='dot_product')
            The type of attention mechanism to use. Options are 'dot_product' (standard attention) or 'expressive' (expressive attention).
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.embedding_dim,
                    n_heads,
                    net_expansion_factor = net_expansion_factor,
                    attention_type = attention_type
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.linear_fc_out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, input, mask = None):
        """
        input : torch.Tensor
            The input tensor of shape (batch_size, seq_length, embedding_dim), where `batch_size` is the number of sequences in a batch,
            `seq_length` is the length of the sequence, and `embedding_dim` is the dimensionality of each token embedding.
        mask : torch.Tensor, optional
            An optional mask tensor to ignore certain positions in the sequence, used for padding or causality.
            Shape should be (batch_size, 1, 1, key_len).

        return : torch.Tensor
            The output tensor of shape (batch_size, seq_length, embedding_dim), containing the processed representation after applying
            the Transformer layers and the final linear projection.
        """

        # ic(input.shape)
        # input = self.one_hot_encode(input)
        #ic(input.shape)

        # input dimensions: batch_size, sequence_len, embedding_dim
        assert (input.dim() == 3),f"Expected input of shape (batch_size, len, embedding_dim) but got shape {input.shape}"
        assert (input.shape[2] == self.embedding_dim), "Transformer input needs to be of size embedding_dim (for each batch)."

        """
        out = self.dropout(
            (self.word_embedding(input) + self.position_embedding(positions))
        )
        """
        data = input.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        for layer in self.layers:
            data = layer(data, mask)
        out = self.norm(self.linear_fc_out(data))
        #ic(out.shape)
        #out = self.one_hot_decode(out)
        #ic(out.shape)
        return out

    def one_hot_encode(self, input):
        """
        Does One-Hot-Codierung for input sequence.
        """
        return F.one_hot(input, num_classes=self.embedding_dim).float()

    def one_hot_decode(self, one_hot_vectors):
        return torch.argmax(one_hot_vectors, dim=-1)



def train(model):
    model.train() # training mode
    for batch_id, (data, target) in enumerate(train_data):
        optimizer.zero_grad()
        out = model(data)
        loss = model.criterion(out, target)
        loss.backward()
        optimizer.step()


def test(model, test_data):
    model.eval() # Gewichte eingefrohren, nicht mehr lernen
    loss = 0
    correct = 0
    for data, target in test_data:
        #print(data.size())

        # Forward, evtl mit alibi
        out = model(data) # alternativ alibi_bias=alibi_bias

        # Loss Berechnung
        loss += F.nll_loss(out, target, reduction='sum').item()

        # Prediction und Accuracy Berechnung
        pred = out.data.max(1, keepdim=True)[1] # 1 -> batch dimension ignorieren
        correct += pred.eq(target.data.view_as(pred)).sum().item() # eq - prüft auf gleichheit

    average_loss = loss / len(test_data.dataset)
    accuracy = 100. * correct / len(test_data.dataset)
    #print('avarage loss: ', loss / len(test_data.dataset))
    #print('accuracy: ', 100. * correct / len(test_data.dataset))
    return accuracy, loss

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = "abcd"
    train_data = ...
    momentum = 0.7
    learning_rate = 0.01
    vocab = "abcdefghijklmnopqrstuvwxyz"
    embedding_dim = len(vocab)
    num_layers = 2
    n_heads = 4
    model = Transformer(
        embedding_dim,
        num_layers,
        n_heads,
        device,
        net_expansion_factor = 4,
    ).to(device)

    criterion = F.nll_loss
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    character_embedding = character_one_hot_embedding(vocab)
    positional_encoding = alibi_positional_encoding(n_heads, len(input))
    character_embedding.encode(input)
