from django.urls import path
from .views import prediction_endpoint

urlpatterns = [
    path('predict/', prediction_endpoint, name='prediction_endpoint'),
]
