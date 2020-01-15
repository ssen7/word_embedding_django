import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
torch.manual_seed(1)


def make_context_vector(inputs, word_to_ix):
    context, target = inputs[0], inputs[1]
    context_idxs = torch.tensor([word_to_ix[w]
                                 for w in context], dtype=torch.long)
    target_idxs = torch.tensor([word_to_ix[target]], dtype=torch.long)
    return context_idxs, target_idxs


class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = sum(self.embeddings(inputs)).view(1, -1)
        out = self.linear1(embeds)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.log_softmax(out, dim=1)
        return out


CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
EPOCHS = 50


def train_model(text, epochs=EPOCHS):

    raw_text = text.split()

    vocab = set(raw_text)
    vocab_size = len(vocab)

    word_to_ix = {word: i for i, word in enumerate(vocab)}

    data = []
    for i in range(2, len(raw_text)-2):
        context = [raw_text[i-2], raw_text[i-1], raw_text[i+1], raw_text[i+2]]
        target = raw_text[i]
        data.append((context, target))

    # print(data[:5])

    model = CBOW(vocab_size, EMBEDDING_DIM)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    losses = []
    for epoch in range(epochs):

        total_loss = 0
        for ins in data:
            context_idxs, target_idxs = make_context_vector(ins, word_to_ix)
            model.zero_grad()
            log_probs = model(context_idxs)
            loss = loss_function(log_probs, target_idxs)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        losses.append(total_loss)
        print('Epoch completed: {}'.format(epoch))

    return model, word_to_ix, data
