from django import forms


class TextForm(forms.Form):

    text = forms.CharField(widget=forms.Textarea)
    epochs = forms.IntegerField(label='Epochs for training', required=True)
    sentence = forms.CharField(label='4 words to predict from')
