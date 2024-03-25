from django.urls import path
from .views import upload_pdf, home

app_name = 'ThesisApp'

urlpatterns = [
    path('', home, name='main'),
    path('upload_pdf/', upload_pdf, name='upload_pdf'),
]