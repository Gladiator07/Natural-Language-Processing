# Importing stuff
from six import Iterator
import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

import spacy
import numpy as np

import random
import math
import time

# Setting SEED
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

spacy_de = spacy.load("de_core_news_sm")
spacy_en = spacy.load("en_core_web_sm")


def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


# Creating fields (to append <sos> and <eos> tag for all the sentences)

SRC = Field(tokenize=tokenize_de, init_token="<sos>", eos_token="<eos>", lower=True)

TRG = Field(tokenize=tokenize_en, init_token="<sos>", eos_token="<eos>", lower=True)

train_data, valid_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(SRC, TRG)
)

print("Example from train data:")
print(vars(train_data.examples[0]))

SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), batch_size=BATCH_SIZE, device=device
)


# Building the Seq2Seq model

# ENCODER
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.embedding = nn.Embedding(
            input_dim, emb_dim
        )  # no dropout, as only one layer

        self.rnn = nn.GRU(emb_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch_size]
        embedded = self.dropout(self.embedding(src))

        # embedded = [src len, batch_size, emb_dim]

        outputs, hidden = self.rnn(embedded)  # no cell state

        # outputs = [src len, batch_size, hid_dim*n_directions]
        # hidden = [n_layers*n_directions, batch_size, hid_dim]
        # outputs are always from the top hidden layer

        return hidden


# DECODER


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.output_dim = output_dim

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim)

        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, context):

        # input = [batch_size]
        # hidden = [n_layers*n_directions, batch_size, hid_dim]
        # context = [n_layers*n_directions, batch_size, hid_dim]

        # n_layers and n_directions in the decoder will both always be 1, therefore;
        # hidden = [1,batch_size, hid_dim]
        # context = [1, batch_size, hid_dim]

        input = input.unsqueeze(0)

        # input = [1, batch_size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch_size, emd_dim]

        emb_con = torch.cat((embedded, context), dim=2)

        # emb_con = [1, batch_size, emb_dim + hid_dim]

        output, hidden = self.rnn(emb_con, hidden)

        # output = [seq_len, batch_size, hid_dim*n_directions]
        # hidden = [n_layers*n_directions, batch_size, hid_dim]

        # seq_len, n_layers and n_directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [1, batch size, hid dim]

        output = torch.cat(
            (embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)), dim=1
        )

        # output = [batch_size, emb_dim + hid_dim * 2]

        prediction = self.fc_out(output)

        # prediction = [batch_size, output_dim]

        return prediction, hidden


# Putting encoder and decoder together


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert (
            encoder.hid_dim == decoder.hid_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):

        # src = [src_len, batch_size]
        # trg = [trg_len, batch_size]

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is the context
        context = self.encoder(src)

        # context also used as the initial hidden state of the decoder
        hidden = context

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden state and the context state
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, context)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as nexr input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs


# TRAINING THE MODEL

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DEC_DROPOUT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Seq2Seq(enc, dec, device).to(device)


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)


model.apply(init_weights)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"The model has {count_parameters(model)}:, trainable parameters")

optimizer = optim.Adam(model.parameters())

TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

# Training loop


def train(model, iterator, optimizer, criterion, clip):

    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]
        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# Evaluate


def evaluate(model, iterator, crietrion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():

        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)  # turn off teacher forcing

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10
CLIP = 1

best_valid_loss = float("inf")

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "tut2-model.pt")

    print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
    print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}")
    print(f"\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}")


model.load_state_dict(torch.load("tut2-model.pt"))

test_loss = evaluate(model, test_iterator, criterion)

print(f"| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |")
