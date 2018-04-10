from django.conf.urls import url
from . import views

urlpatterns = [
	url('about/', views.about, name='generation/about'),
	url('demo/', views.demo, name='generation/demo'),
	url('howtouse/', views.howtouse, name='generation/howtouse'),
]