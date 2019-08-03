import os
import pickle
import cv2
from os.path import isfile,join
import imutils
import moviepy.editor as mp

fileNames=[]
for filename in os.listdir("D:/UCF11/data"):
	for filename2 in os.listdir("D:/UCF11/data/"+filename):
		for filename3 in os.listdir("D:/UCF11/data/"+filename+'/'+filename2):
			fileNames.append([filename3,"_".join(filename3.split("_")[1:-2])]) 
fourcc = cv2.VideoWriter_fourcc(*"DIVX")
cap = cv2.VideoCapture("D:/UCF11/data/basketball/v_shooting_01/v_shooting_01_01.avi")
out = cv2.VideoWriter("D:/UCF11/data/basketball/v_shooting_01/v_shooting_01_01_edit.avi", fourcc, 30.0, (176,144), True)
print(cap.get(cv2.CAP_PROP_FPS))
while(cap.isOpened()):
	ret, frame = cap.read()
	if ret == False:
		break
	else:
		cv2.imshow('frame',frame)
		imutils.resize(frame, width =176, height = 144)
		input("press enter to continue")
		cv2.imshow('newframe',frame)
		out.write(frame)
		if (cv2.waitKey(1) & 0xFF == ord('q')):
			break
		
		

			

			#ffmpeg -i "D:/UCF11/data/"+filename+'/'+filename2+'/'+filename3 -vf scale=176:144 "D:/UCF11/new_data/"+filename+'/'+filename2+'/'+filename3			
			
with open("fileName.pickle", 'wb') as handle:
	pickle.dump(fileNames,handle)

