from django.conf.urls import url
from . import views

urlpatterns = [
    url('drgsreviews/get_data/', views.get_data, name='get_data'),
	url('drgsreviews/execute/', views.execute, name='execute'),
    
]
