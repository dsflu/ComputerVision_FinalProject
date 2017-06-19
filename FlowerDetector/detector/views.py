#!/usr/bin/env python
# -*- coding: latin-1 -*-
#CODED BY XIANGWEI SHI
from django.shortcuts import render
from django.http import HttpResponse
from django.template import RequestContext
from FlowerDetector.bounding_box import bounding_box
import os

# Create your views here.
def index(request):
    return render(request,'index.html')#request the homepage

def upload_image(request):
	if request.method == 'POST':
		message = {}
		myFile=request.FILES.get("myfile",None)
		if not myFile:
			return HttpResponse("No files for upload!")
		file_path=os.path.join("E:\sxw\\tudelft\course\Quarter 4\IN4393 Computer Vision\project\interface\FlowerDetector\detector\static", myFile.name)
		destination=open((file_path),'wb+')
		for chunk in myFile.chunks():
			destination.write(chunk)
		destination.close()
		message['file_path']=str(myFile.name)
		bounding_box(str(file_path))
		return  render(request,'return.html',message)
