from django.shortcuts import render, redirect
from .forms import TextForm
from .embeddings import train_model, make_context_vector
import numpy as np


def get_text(request):

    if request.method == 'POST':
        form = TextForm(request.POST)
        if form.is_valid():
            text = form.cleaned_data['text']
            data_val = form.cleaned_data['data_val']
            epochs = form.cleaned_data['epochs']

            model, word_to_ix, data = train_model(text, epochs)

            ix_to_word = {word_to_ix[key]: key for key in word_to_ix.keys()}

            x = data[data_val]

            context, target = make_context_vector(x, word_to_ix)

            predicted = ix_to_word[np.argmax(model(context).data.numpy())]

            target = ix_to_word[target.numpy()[0]]
        return render(request, 'form.html', {'form': form, 'predicted': predicted, 'target': target})

    form = TextForm()
    return render(request, 'form.html', {'form': form})
