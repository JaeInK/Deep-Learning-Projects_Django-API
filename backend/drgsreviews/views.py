from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .scripts import drugsreviews as dr
import json


# Create your views here.
@csrf_exempt
def get_data(request):
	dic = {}
	dic['test_user'], dic['test_drug'], dic['test_review'], dic['test_sentiment'] = dr.get_data()  
	dic['test_sentiment'] = int(dic['test_sentiment'])

	return JsonResponse(dic)

@csrf_exempt
def execute(request): 
	print('--------------------------------------------')
	dic = json.loads(request.body.decode('utf-8'))
	dic.pop('predict', None)
	if dic['test_sentiment'] is None:
		dic['test_sentiment'] = 100

	print(dic)
	
	# tf.keras.backend.clear_session()
	result = dr.execute_model(dic)
	res = {}
	res['predict'] = int(result[1])
	print('result-----')
	print(result[1])
	return JsonResponse(res)		
