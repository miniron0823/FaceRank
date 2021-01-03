from django.shortcuts import render
from django.http import HttpResponse
import simplejson as json


def index(request):

    
    return render(request, 'appFaceRank/index.html')

    #return HttpResponse("Hello, world. You're at the polls index.")
    
def getPicture(request):
    context = {"hellow": "imhellow"}
    return HttpResponse(json.dumps(context), content_type="application/json")