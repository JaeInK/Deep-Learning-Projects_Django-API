from django.shortcuts import render
from .forms import GenerationForm

import requests
import json

def demo(request):
	if request.method == "POST":
		form = GenerationForm(request.POST)
		if form.is_valid():
			data = form.cleaned_data
			dic ={}
			dic['word'] = data['word']
			print(dic)
			res = requests.get('http://165.132.106.71:7200/generation/run/', data=json.dumps(dic))
			print(res)
			res = res.json()
			print(res['sentence'])
			form = GenerationForm(initial={'word': dic['word'], 'output': res['sentence']})
	else:
		form = GenerationForm()
	return render(request,'generation/demo.html', {'form': form})


def about(request):
	return render(request, 'generation/about.html')

def howtouse(request):
	return render(request, 'generation/howtouse.html')