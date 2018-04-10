from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from .scripts import utils, cnn_lstm, cnn_cnn, han, pa

import requests
import json
import random

# Create your views here.

@csrf_exempt
def get_data(request):
    review_text = list(open("mvreviews/data/reviews.txt", "r").readlines())
    products = list(open("mvreviews/data/titles.txt", "r").readlines())
    labels = list(open("mvreviews/data/labels.txt", "r").readlines())
    idx = random.randrange(0, len(review_text) - 1)

    review = utils.s_to_dot(review_text[idx])
    product = products[idx]
    label = labels[idx]

    res_dict = {'review': review, 'product':product, 'label' : label}
    return JsonResponse(res_dict)

@csrf_exempt
def run(request):
    received_data = json.loads(request.body.decode('utf-8'))
    review = received_data['review']
    movie_id = received_data['movie_id']
    review_s = utils.punct_to_s(review)
    
    f = open("mvreviews/data/test/test.txt", "w")
    f.write(review_s + "\n")
    f.close()

    f = open("mvreviews/data/test/product_nsc.txt", 'w')
    f.write(movie_id + "\n")
    f.close()

    model_cnn_lstm = cnn_lstm.CNN_LSTM()
    model_cnn_cnn = cnn_cnn.CNN_CNN()
    model_han = han.HAN()
    model_pa = pa.PA()

    pred_cnn_lstm = int(model_cnn_lstm.predict()) ## return numpy.int64 -> int()
    pred_cnn_cnn = int(model_cnn_cnn.predict())
    pred_han = int(model_han.predict())
    pred_pa = int(model_pa.predict())

    res_dict = {'pred_cnn_lstm': pred_cnn_lstm, 'pred_cnn_cnn':pred_cnn_cnn, 'pred_han':pred_han, 'pred_pa':pred_pa}
    return JsonResponse(res_dict)
