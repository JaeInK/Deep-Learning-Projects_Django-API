from django.db import models
from django import forms

class BaseForm(forms.Form):
    review = forms.CharField(widget = forms.Textarea(attrs={'cols':'50', 'rows':'10'}), max_length = 5000)
    moview_id = forms.CharField(disabled=True)
    y_label = forms.IntegerField(disabled=True)
