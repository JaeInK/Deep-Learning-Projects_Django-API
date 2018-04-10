from django.conf.urls import url, include
from . import views

urlpatterns = [
    # url('networks2/', views.networks_backup, name='networks'),
    url('mvreviews/get_data/', views.get_data, name='get_data'),
    url('mvreviews/run/', views.run, name='run'),

]
