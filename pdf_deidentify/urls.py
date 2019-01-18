from django.urls import path
from . import views
from django.views.generic import View

urlpatterns = [
	path('list', views.doclist, name='doclist'),
	path('', views.homeview, name='homeview')
]