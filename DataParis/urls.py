from django.contrib import admin
from django.urls import path
from . import views


# app_name = 'DataParis'

urlpatterns = [
    path("", views.home, name="home"),
    path("home/", views.home, name="home"),
    path("question1/", views.question1, name="question1"),
    path("question2/", views.question2, name="question2"),
    path("question3/", views.question3, name="question3"),
    path("map/", views.map, name="map"),
]
