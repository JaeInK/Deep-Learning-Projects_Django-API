from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

import json
from .scripts.generate import main

@csrf_exempt
def generate(request):
	print('generate--------')
	dic = json.loads(request.body.decode('utf-8'))
	print(dic)
	dic['sentence'] = main(dic['word'])
	print(dic['sentence'])
	return JsonResponse(dic)
