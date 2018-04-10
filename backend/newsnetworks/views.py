from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from backend.settings import NEWS_DIR

# from .scripts import newsnetworks_web

import requests
import pandas as pd
import networkx as nx

import os
import sys
import json
import time

sys.path.append(os.path.join(NEWS_DIR,'scripts/'))

NET_DIC = {'child' : '소년법', 'coin': '가상화폐', 'clear': '적폐청산', 'cortax': '법인세', 'estate': '부동산', 'fulltime': '정규직', 'korea': '남북관계', 'macro': '거시경제', 'nuclear': '원자력발전소', 'wage': '최저임금'}
# Create your views here.
# @csrf_exempt
# def networks_backup(request):
#     received_data = json.loads(request.body.decode('utf-8'))
#     dataset = received_data['dataset']
#     edge_th = received_data['edge_threshold']
#     deg_th = received_data['degree_threshold']
#     max_sub_flag = received_data['max_subgraph']
#     net_name = NET_DIC[dataset]
#
#     pickle_path = os.path.join(pwd + '/data/' + str(dataset) + '_extracted_with_polarity.p')
#     st = time.time()
#     df = pd.read_pickle(pickle_path)
#     print("read_pickle opt elapsed: {} secons".format(time.time()-st))
#
#     G = nnw.NewsNetwork()
#     st = time.time()
#     G.read_file(df)
#     print("read_file opt elapsed: {} secons".format(time.time()-st))
#     st = time.time()
#     G.pre_processing(edge_threshold = edge_th, deg_threshold = deg_th)
#     print("pre_processing opt elapsed: {} secons".format(time.time()-st))
#     st = time.time()
#     if max_sub_flag:
#         max_sub = G.subgraph(max_subgraph = True)
#         G.set_node_link_attrs(max_sub)
#
#     else:
#         G.set_node_link_attrs(edge_bold = 10)
#     print("set_node_link_attrs opt elapsed: {} secons".format(time.time()-st))
#     st = time.time()
#     G_dict = G.get_node_link_data()
#     print("get_node_link_data opt elapsed: {} secons".format(time.time()-st))
#     st = time.time()
#     network_info = G.get_network_attrs(net_name)
#     print("get_network_attrs opt elapsed: {} secons".format(time.time()-st))
#     G_dict.update(network_info)
#
#     return JsonResponse(G_dict)

@csrf_exempt
def networks(request):
    received_data = json.loads(request.body.decode('utf-8'))
    dataset = received_data['dataset']
    edge_th = received_data['edge_threshold']
    deg_th = received_data['degree_threshold']
    max_sub_flag = received_data['max_subgraph']
    net_name = NET_DIC[dataset]

    pickle_path = os.path.join('newsnetworks/data/' + str(dataset) + '.pickle')
    st = time.time()
    G = pd.read_pickle(pickle_path)
    print("read_pickle opt elapsed: {} secons".format(time.time()-st))

    G.pre_processing(edge_threshold = edge_th, deg_threshold = deg_th)
    print("pre_processing opt elapsed: {} secons".format(time.time()-st))
    st = time.time()
    if max_sub_flag:
        max_sub = G.subgraph(max_subgraph = True)
        G.set_node_link_attrs(max_sub)

    else:
        G.set_node_link_attrs(edge_bold = 10)
    print("set_node_link_attrs opt elapsed: {} secons".format(time.time()-st))
    st = time.time()
    G_dict = G.get_node_link_data()
    print("get_node_link_data opt elapsed: {} secons".format(time.time()-st))
    st = time.time()
    network_info = G.get_network_attrs(net_name)
    print("get_network_attrs opt elapsed: {} secons".format(time.time()-st))
    G_dict.update(network_info)

    return JsonResponse(G_dict)
