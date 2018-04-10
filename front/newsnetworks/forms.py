from django import forms

class NewsForm(forms.Form):
    # (value, name to visualize)
    DATASET_CHOICES = (('child', '소년법'), ('coin', '가상화폐'), ('clear', '적폐청산'), ('cortax', '법인세'), ('estate', '부동산'), ('fulltime', '정규직'), ('korea', '남북관계'), ('macro', '거시경제'), ('nuclear', '원자력발전소'), ('wage', '최저임금'))
    
    dataset = forms.ChoiceField(widget=forms.Select(attrs={'class':'form-control'}), choices = DATASET_CHOICES)
    edge_threshold = forms.IntegerField(widget=forms.NumberInput(attrs={'class':'form-control'}), initial = 2, max_value = 10, min_value = 0)
    degree_threshold = forms.IntegerField(widget=forms.NumberInput(attrs={'class':'form-control'}), initial = 1 ,max_value = 10, min_value = 0)
    # max_subgraph = forms.BooleanField(widget=forms.CheckboxInput)
    max_subgraph = forms.BooleanField(widget=forms.CheckboxInput(attrs={'class':'form-check-input'}), initial = False, required=False)