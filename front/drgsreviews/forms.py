from django import forms

class DrgsForm(forms.Form):
	user = forms.CharField(widget=forms.TextInput(attrs={'class':'form-control', 'placeholder':'Get Random User/Drug'}),max_length=200)
	drug = forms.CharField(widget=forms.TextInput(attrs={'class':'form-control', 'placeholder':'Get Random User/Drug'}),max_length=200)
	review = forms.CharField(widget=forms.Textarea(attrs={'class':'form-control', 'id': 'Textarea', 'cols':'50', 'rows': '10', 'placeholder': '마침표가 있는 영문 문장으로 입력하세요.'}),max_length =5000)
	#review = forms.CharField(widget = forms.Textarea(attrs={'cols':'50', 'rows':'10'}), max_length = 5000)
	sentiment = forms.IntegerField(widget=forms.NumberInput(attrs={'class':'form-control'}),required=False)
	predict = forms.IntegerField(widget=forms.NumberInput(attrs={'class':'form-control'}),required=False)
