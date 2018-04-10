from django import forms


class MvreviewsForm(forms.Form):
    review = forms.CharField(widget = forms.Textarea(attrs={'class': 'form-control', 'cols':'50', 'rows':'10'}), max_length = 5000, disabled=True)
    
    # movie_id = forms.CharField(disabled=True, initial='hi')
    # y_label = forms.IntegerField(disabled=True, initial = 1)
    movie_id = forms.CharField(widget=forms.TextInput(attrs={'class':'form-control'}), required = False, disabled=True)
    y_label = forms.IntegerField(widget=forms.NumberInput(attrs={'class':'form-control'}), required = False, disabled=True)





