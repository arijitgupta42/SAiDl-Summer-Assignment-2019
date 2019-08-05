from __future__ import print_function, division
import cv2
import os
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class RandomCrop(object):
    

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, clip):

        h, w = clip.size()[2:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        clip = clip[:, :, top: top + new_h,
               left: left + new_w]

        return clip


class videoDataset(Dataset):
    
    def __init__(self, clipsListFile="D:/UCF11/fileName.pickle", rootDir="D:/UCF11/data", channels=3, timeDepth=1, xSize=144, ySize=176, mean=[0.485, 0.456, 0.406], transform=None):
       
        with open(clipsListFile, "rb") as fp:  
            clipsList = pickle.load(fp)

        self.clipsList = clipsList
        self.rootDir = rootDir
        self.channels = channels
        self.timeDepth = timeDepth
        self.xSize = xSize
        self.ySize = ySize
        self.mean = mean
        self.transform = transform


    def __len__(self):
        return len(self.clipsList)

    def readVideo(self, videoFile):
  
        cap = cv2.VideoCapture(videoFile)

        frames = torch.FloatTensor(self.channels, self.timeDepth, self.xSize, self.ySize)
        failedClip = False
        for f in range(self.timeDepth):

            ret, frame = cap.read()
            if ret:
                frame = torch.from_numpy(frame)
            
                frame = frame.permute(2, 0, 1)
                frames[:, f, :, :] = frame

            else:
                print("Skipped!")
                failedClip = True
                break

        for c in range(3):
            frames[c] -= self.mean[c]
        frames /= 255
        return frames, failedClip

    def __getitem__(self, idx):

        videoFile = os.path.join(self.rootDir, self.clipsList[idx][0])
        clip, failedClip = self.readVideo(videoFile)
        if self.transform:
            clip = self.transform(clip)
        sample = {'clip': clip, 'label': self.clipsList[idx][1], 'failedClip': failedClip}

        return sample


