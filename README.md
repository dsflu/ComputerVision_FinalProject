# ComputerVision_FinalProject

### Authors
Made by  [Feng Lu](https://github.com/fredericklu) and Xiangwei Shi for the IN4393: Computer Vision Project

#### This is a flower species recognizing system based on 'bag of visual word' algorithm with an intereface.
This file is built for interpretation.
Before you download and use these codes. You need to know that this system is built in Python2 with OpenCV3 package, scikit-learn package and Django package(0.11).
You can install the environment and the packages if you need.

### Requirement:

1. The enviornment and the packages mentioned above
2. After downloading the codes, go into the directory of FlowerDetector/detector and open the file of 'views.py'. Change the addresses as instructed.
3. Go into the directory of FlowerDetector/FlowerDetector. Open the files of 'bow.py' and 'detection.py'. Also change the addresses as instructed.

### Instruction for the user interface:

1. Open a shell or command prompt
2. Go into the directory of FlowerDetector (your/path/to/FlowerDetector)
3. Type in 'python manage.py runserver'
4. Copy the url in your command and log on the localhost with that url (mostly the url is 127.0.0.1:8000)

#### About the user interface:
1. There are three different kinds of interfaces.
2. After you log on the localhost, you will see the homepage of user interface. There are two buttons under a background image. The first one is to select an image that you want to recognize. The second button is a 'submit' button.
3. Once submitting a testing image, you can see a new interface with one image and one 'detect' button. The image shows a bounding box on your testing image, which detects the flower.
4. The last kind of interface can show much informaction, such as the detected label with the highest confidence, a reference image from the dataset with the same label as the detected one, two other buttons for reuploading another test image and other information.

#### One thing needs to be mentioned: The testing image of flower needs to be taken in close range. And the fewer flowers appearing in one image, the better result can be obtained.
