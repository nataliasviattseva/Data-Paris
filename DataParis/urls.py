from django.contrib import admin
from django.urls import path
from . import views


# app_name = 'DataParis'

urlpatterns = [
    path("", views.home, name="home"),
    path("home/", views.home, name="home"),
    path("Question1.html", views.Question_1, name="question1"),
    path("question2/", views.question2, name="question2"),
    path("Question3.html", views.Question_3, name="question3"),
]
