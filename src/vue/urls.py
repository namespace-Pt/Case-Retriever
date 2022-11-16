from django.urls import path

from . import views

urlpatterns = [
    path('', views.main, name='vue-main'),
    path('detail/<str:id>/', views.detail, name='vue-detail'),
    path('download/<str:id>/', views.download, name='vue-download'),
    path('explain/', views.explain, name='vue-explain'),
    path('debug/', views.debug, name='vue-debug'),
]