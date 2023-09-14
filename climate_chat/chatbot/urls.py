from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from . import views

app_name = 'image_classification'
urlpatterns = [
    # two paths: with or without given image
    path('', views.index, name='index'),
    path('api/', views.BasicApi.as_view(), name='api')
]
