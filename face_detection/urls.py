from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('video_stream/', views.video_stream, name='video_stream'),
    path('get_questions/', views.get_questions, name='get_questions'),
] 