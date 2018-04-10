from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.generic import TemplateView

from .forms import MvreviewsForm
from . import forms
# Create your views here.

import requests
import json


class Mvreviews(TemplateView):
    template_name = 'mvreviews/demo.html'

    def get(self, request):
        form = MvreviewsForm()
        return render(request, self.template_name, {'form': form})

    def post(self, request):
        form = MvreviewsForm(request.POST)
        
        if form.is_valid():
            review = form.cleaned_data['review']
            movie_id = form.cleaned_data['movie_id']
            y_label = form.cleaned_data['y_label']
            payload = json.dumps({'review':review, 'movie_id':movie_id})
            print(payload)
            res = requests.post("http://165.132.106.71:7200/mvreviews/run/", data = payload)

            res_json = res.json() ## all predicted value are 'int'

            # form = MvreviewsForm(initial={'review':review, 'movie_id':movie_id, 'y_label':y_label})
            
            return render(request, self.template_name, {'form':form, 'y_label': y_label, 'res':res_json})
        else:
            
            return render(request, self.template_name, {'form':form, 'res':'ERROR'})


@csrf_exempt
def get_data(request):
    res = requests.post("http://165.132.106.71:7200/mvreviews/get_data/")
    res_json = res.json()
    review = res_json['review']
    product = res_json['product']
    label = int(res_json['label'])
    
    return JsonResponse ({'review':review, 'product':product, 'label':label})

def about(request):
    return render(request, 'mvreviews/about.html')
    
def howtouse(request):
    return render(request, 'mvreviews/howtouse.html')