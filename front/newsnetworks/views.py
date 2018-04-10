from django.shortcuts import render
from django.views.decorators.csrf import csrf_protect
from django.views.generic import TemplateView
from .forms import NewsForm

import requests
import json

# Create your views here.
class NewsNetworksView(TemplateView):
    template_name = 'newsnetworks/demo.html'

    def get(self, request):
        form = NewsForm()
        return render(request, self.template_name, {'form': form})

    def post(self, request):
        form = NewsForm(request.POST)
        if form.is_valid():
            dataset = form.cleaned_data['dataset']
            edge_th = form.cleaned_data['edge_threshold']
            degree_th = form.cleaned_data['degree_threshold']
            max_subgraph = form.cleaned_data['max_subgraph'] # send boolean, True, False

            payload = json.dumps({'dataset':dataset, 'edge_threshold':edge_th, 'degree_threshold':degree_th, 'max_subgraph':max_subgraph})
            res = requests.post('http://165.132.106.71:7200/newsnetworks/get_data/', data=payload)
            g_json = json.dumps(res.json())
            return render(request, 'newsnetworks/networks.html', {'netdata_json':g_json})

        else:
            return render(request, self.template_name, {'form':form})

def about(request):
    return render(request, 'newsnetworks/about.html')

def howtouse(request):
    return render(request, 'newsnetworks/howtouse.html')
