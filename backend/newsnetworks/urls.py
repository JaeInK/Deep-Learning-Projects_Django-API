from django.conf.urls import url, include
from . import views

urlpatterns = [
    # url('networks2/', views.networks_backup, name='networks'),
    url('newsnetworks/get_data/', views.networks, name='networks'),
    
]
