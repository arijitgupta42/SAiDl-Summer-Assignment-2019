import os
import pickle
import cv2
from os.path import isfile,join
import imutils
import moviepy.editor as mp
fileNames=[]
rep = input("Do you want to resize videos?(y/n)")
for filename in os.listdir("D:/UCF11/data"):
	for filename2 in os.listdir("D:/UCF11/data/"+filename):
		for filename3 in os.listdir("D:/UCF11/data/"+filename+'/'+filename2):
			if(filename3.split(".")[0].split("_")[-1] == 'resized'):
				fileNames.append([filename3,"_".join(filename3.split("_")[1:-3])])
				 
			if(rep == 'y'):
				import moviepy.editor as mp
				clip = mp.VideoFileClip("D:/UCF11/data/"+filename+'/'+filename2+'/'+filename3)
				clip_resized = clip.resize( (176,144) )
				print(clip_resized.size)
				clip_resized.write_videofile("D:/UCF11/data/"+filename+'/'+filename2+'/'+filename3.split(".")[0]+"_resized.avi",codec = 'h264')
				print("Successfully Resized video %d", n)
				clip.reader.close()
			
with open("fileName.pickle", 'wb') as handle:
	pickle.dump(fileNames,handle)


