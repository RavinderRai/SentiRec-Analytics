from django.urls import include, path
from . import views

urlpatterns = [
    path('<int:headphone_id>/', views.headphone_detail, name='headphone_detail'),
]