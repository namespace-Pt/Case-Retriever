from django.urls import path

from . import views

urlpatterns = [
    path('', views.main, name='search-main'),
    path('detail/<str:id>/', views.detail, name='search-detail'),
    path('download/<str:id>/', views.download, name='search-download'),
    path('explain/', views.explain, name='search-explain'),
    path('debug/', views.debug, name='search-debug'),
]