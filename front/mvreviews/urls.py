from django.conf.urls import url
from .views import Mvreviews
from . import views

urlpatterns = [
    url('about/', views.about, name = 'mvreviews/about'),
    url('demo/', Mvreviews.as_view(), name='mvreviews/demo'),
    url('get_data/', views.get_data, name='mvreviews/get_data'),
    url('howtouse/', views.howtouse, name='mvreviews/howtouse'),
]
