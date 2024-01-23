from django.shortcuts import render
from .models import Headphone

def headphone_detail(request, headphone_id):
    headphone = Headphone.objects.get(pk=headphone_id)
    return render(request, 'analytics/headphone_detail.html', {'headphone': headphone})


