from django.urls import path

from . import views

urlpatterns = [
    path('', views.main, name='search-main'),
    path('detail/<str:id>/', views.detail, name='search-detail'),
    path('debug/', views.debug, name='search-debug'),
]