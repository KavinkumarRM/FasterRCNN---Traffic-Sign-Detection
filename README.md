# Traffic-Sign-Detection---computer-vision
This is my accomplishment on Computer vision and Real time object detection. I have used Faster R-CNN for object detection. 

There are two sections in the attached script,

1) Computing on the AWS server.
	The scripts in AWS is the same as in the local machine but the only difference is that there are two pairs of command shell script employed for the process.

	=>The Master.command script to initiate the execution of the video.py script. 
	
	=>video.py script pushes the video into AWS and gets back the signal for the completion of the process
	
	=>script in AWS (EC2 p2.xlarge ubuntu instance) completes the process then send the predicted frames and which is again covered into videos locally.

The scripts provided in the aws_script folder is the one with which the python frcnn framework is executed. 

2) Computing in the local machine.
	The script master_local.command can execute the file in the local machine. It executed video_local.py which initiates the process of executing the file locally. This file initiates the process by calling the execute file and initiating the frcnn framework. 

Steps to execute locally:

1) Put the video in this folder named as video.avi 

2) Double click master_local.command. 

Starts the whole execution which would take a while depending upon the file size. The output would be saved in the same folder as final_output.avi


Package needed.

h5py
 | Keras==2.0.3
 | numpy
 | opencv-python
 | sklearn
 | Subprocess
 | PIL
