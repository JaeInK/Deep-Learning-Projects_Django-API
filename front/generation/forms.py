from django import forms

class GenerationForm(forms.Form):
	word = forms.CharField(widget=forms.TextInput(attrs={'class':'form-control', 'placeholder':'한 개의 단어를 입력하세요.'}), max_length=200)
	output = forms.CharField(widget = forms.Textarea(attrs={'class':'form-control', 'cols':'50', 'rows':'10'}), max_length=2000, required = False)
