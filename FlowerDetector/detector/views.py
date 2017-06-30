#!/usr/bin/env python
# -*- coding: latin-1 -*-
#CODED BY XIANGWEI SHI
from django.shortcuts import render
from django.http import HttpResponse
from django.template import RequestContext
from FlowerDetector.bounding_box import bounding_box
from FlowerDetector.detection import detection
import pandas as pd
import os

# Create your views here.
message = {}
address = "/Users/fredlu/Developer/ComputerVision/FlowerDetector/detector/static/"

def index(request):
    return render(request,'index.html')#request the homepage
def upload_image(request):
	if request.method == 'POST':
		myFile=request.FILES.get("myfile",None)
		rootdir = address
		for root, dirs, files in os.walk(rootdir):
			for file in files:
				if file != 'background.jpg':
					os.remove(os.path.join(address, file))
		if not myFile:
			return HttpResponse("No files for upload!")
		file_path=os.path.join(address, myFile.name)
		destination=open((file_path),'wb+')
		for chunk in myFile.chunks():
			destination.write(chunk)
		destination.close()
		message['file_path']=file_path
		message['file_name']='example'
		bounding_box(address, str(myFile.name))
		return  render(request,'return.html', message)
def detect_image(request):
	if request.method == 'GET':
		rootdir = address
		for root, dirs, files in os.walk(rootdir):
			for file in files:
				if file != 'background.jpg' and file != 'example.jpg':
					file_path = os.path.join(address, file)
					break
			break
		message['prediction'], message['confidence'] = detection(file_path)
		number = int(message['prediction'])
		# return HttpResponse(number)
		labels_data = pd.read_csv('/Users/fredlu/Developer/ComputerVision/csv/label.csv',header=None)
        label = labels_data[labels_data[0] == number]
        idx = label.index[0]+1
        file_path = "/Users/fredlu/Developer/ComputerVision/training_data/"
        file_name = "image_"+'%0*d' % (5, idx) + '.jpg'
        message['file_name'] = file_name
        message['file_path'] = os.path.join(file_path, file_name)
        open(os.path.join(address, file_name),"wb").write(open(message['file_path'],"rb").read())

        return render(request,'detection.html',message)
        
		# 