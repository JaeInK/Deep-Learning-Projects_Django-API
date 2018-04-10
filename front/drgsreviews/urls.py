from django.conf.urls import url
from . import views
from .views import Drgsreviews

urlpatterns = [
	# url('demo/', views.demo, name='drgsreviews/demo'),
	url('demo/', Drgsreviews.as_view(), name='drgsreviews/demo'),
	url('get_data/', views.get_data, name = 'drgsreviews/get_data'),
	url('about/', views.about, name='drgsreviews/about'),
	url('howtouse/', views.howtouse, name='drgsreviews/howtouse'),
]
