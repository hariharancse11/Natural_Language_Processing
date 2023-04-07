from django.shortcuts import render,HttpResponse,redirect
from django.contrib import messages
from .NewsCategorizer import autoCategorize

# Create your views here.
def index(request):
    return render(request,'index.html')

def predict(request):
    
    text = request.GET['newsText']
    res = autoCategorize(text)
    messages.info(request,res)
    return redirect('/')