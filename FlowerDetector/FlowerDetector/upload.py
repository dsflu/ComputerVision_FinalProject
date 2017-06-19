import os

def upload_image(request):
	if request.method == 'POST':
		myFile=request.FILES.get("myfile",None)
		if not myFile:
			return HttpResponse("No files for upload!")
		destination=open(os.path.join("E:\\upload", myFile.name),'wb+')
		for chunk in myFile.chunks():
			destination.write(chunk)
		destination.close()
		ret  HttpResponse("Upload over!")
