from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('video_stream/', views.video_stream, name='video_stream'),
    path('get_current_question/', views.get_current_question, name='get_current_question'),
    path('next_question/', views.next_question, name='next_question'),
] 