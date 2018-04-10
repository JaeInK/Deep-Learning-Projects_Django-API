from django.conf.urls import url
from . import views

urlpatterns = [
	url('generation/run/', views.generate, name='generation'),
]
