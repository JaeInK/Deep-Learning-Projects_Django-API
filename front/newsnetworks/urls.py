from django.conf.urls import url
from .views import NewsNetworksView
from . import views

urlpatterns = [
    url('demo/', NewsNetworksView.as_view(), name='newsnetworks/demo'),
    url('about/', views.about, name='newsnetworks/about'),
    url('howtouse/', views.howtouse, name='newsnetworks/howtouse')
]
