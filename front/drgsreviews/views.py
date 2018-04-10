from django.shortcuts import render
from .forms import DrgsForm
import requests
from django.http import JsonResponse
import json
from django.views.decorators.csrf import csrf_exempt
from django.views.generic import TemplateView

# Create your views here.
class Drgsreviews(TemplateView):
	template_name = 'drgsreviews/demo.html'

	def get(self, request):
		form = DrgsForm()
		
		return render(request, self.template_name, {'form': form})
	
	def post(self, request):
		form = DrgsForm(request.POST)
		if form.is_valid():
			
			data = form.cleaned_data
			dic={}
			dic['test_drug'] = data['drug']
			dic['test_sentiment'] = data['sentiment']
			dic['test_review'] = data['review']
			dic['test_user'] = data['user']
			dic['predict'] = data['predict']

			res = requests.post('http://165.132.106.71:7200/drgsreviews/execute/', data=json.dumps(dic))

			res = res.json()
			
			form = DrgsForm(initial={'user': dic['test_user'], 'drug': dic['test_drug'], 'review': dic['test_review'],
			'sentiment': dic['test_sentiment'], 'predict': res['predict']})

			return render(request, self.template_name, {'form':form})
		else:

			return render(request, self.template_name, {'form':form, 'warn':'User와 Drug를 채워주세요.'})
	
@csrf_exempt
def get_data(request):
	#name = request.GET.get('name', False)
	
	print('send data---------------')
	res = requests.post('http://165.132.106.71:7200/drgsreviews/get_data/')
	print(res.json())
	return JsonResponse(res.json())
	
def about(request):
	return render(request, 'drgsreviews/about.html')

def howtouse(request):
    return render(request, 'drgsreviews/howtouse.html')
