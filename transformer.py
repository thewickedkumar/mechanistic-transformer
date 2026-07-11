
import torch
import torch.nn as nn
import math

# You are getting a batch of tokens so (batch,seq_len); hopefully you send this dim
# Then output size of this func is (batch,seq_len, d_model)
class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model # dimension of token embeddingg
        self.vocab_size = vocab_size  # number of unique tokens
        self.embedding = nn.Embedding(vocab_size, d_model)

        def forward(self, x):
            return self.embedding(x) * math.sqrt(self.d_model) # scaling helps normality and making sure that the positional encoding info and embedding info is comparable

# This function's forward pass takes that batch,n_tokens,d_model as input and outputs same
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_len # number of tokens in the sequence or batch of token
        self.dropout = nn.Dropout(dropout) #Regularization to prevent overfitting
        
        #Creating a matrix of shape (seq legth, d_model)
        pe = torch.zeros(seq_len, d_model) # For each position of seq, get d_model dimvec

        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1) #[dim, 1]

        # Geeky stuff to create positional embeddings to even and odd cols
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)

        pe = pe.unsqueeze(0) # (1, seq_length, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        #x's first dimension has sequence_length data which we exactly need to splice pe
        x = x + (self.pe[:, :x.shape[1], :]).reqires_grad_(False) # Broadcasting 0 dim such that all batches can use the same (1,seq_length, d_model) positional embedding !
        return self.dropout(x)

#goal is to take the batches values and calculate their mean and std dev and kind of regularize them by doing x_new = (x_curr - x_mean)/sqrt(std.dev+eps); eps exists for avoiding small devs to explode x_new;
class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None: # -> None is a type hint
        super().__init__()
            self.eps = eps # numerical stab, and also to avoid division by zero
            self.alpha = nn.Parameter(torch.ones(1)) #nn.Param makes it learnable
            self.bias = nn.Parameter(torch.ones(1)) # This param is added and above mul

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1,keepdim=True)

        return self.alpha * (x - mean) / (std+self.eps) + self.bias
