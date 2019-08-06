import matplotlib.pyplot as plt
import numpy

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from videoDataset import videoDataset

import os

import time

import pickle

import cv2
from os.path import isfile,join

VideoLoad = videoDataset()

n = 0

with open("D:/UCF11/fileName.pickle", "rb") as fp:  
           clipsList = pickle.load(fp) 
length = len(clipsList)

print(clipsList)
labellist=[]

for i in range(length):
    if(clipsList[i][1] == 'shooting'):
        clipsList[i][1] = 0
    if(clipsList[i][1] == 'biking'):
        clipsList[i][1] = 1
    if(clipsList[i][1] == 'diving'):
        clipsList[i][1] = 2
    if(clipsList[i][1] == 'golf'):
        clipsList[i][1] = 3
    if(clipsList[i][1] == 'riding'):
        clipsList[i][1] = 4
    if(clipsList[i][1] == 'juggle'):
        clipsList[i][1] = 5
    if(clipsList[i][1] == 'swing'):
        clipsList[i][1] = 6
    if(clipsList[i][1] == 'tennis'):
        clipsList[i][1] = 7
    if(clipsList[i][1] == 'jumping'):
        clipsList[i][1] = 8
    if(clipsList[i][1] == 'spiking'):
        clipsList[i][1] = 9
    if(clipsList[i][1] == 'walk_dog'):
        clipsList[i][1] = 10
    labellist.append(clipsList[i][1])

train_data = []
test_data = []


 


for filename in os.listdir("D:/UCF11/data"):
    for filename2 in os.listdir("D:/UCF11/data/"+filename):
        for filename3 in os.listdir("D:/UCF11/data/"+filename+'/'+filename2):
            if(filename3.split(".")[0].split("_")[-1] == 'resized'):
                    sample = videoDataset.readVideo(VideoLoad, "D:/UCF11/data/"+filename+'/'+filename2+'/'+filename3)
                    
                    if(n%3==0):
                        train_data.append([sample[0], torch.from_numpy(numpy.array(labellist[n]))])
                        print(sample[0].shape)
                                                
                    else:
                        test_data.append([sample[0], torch.from_numpy(numpy.array(labellist[n]))])
                        print(sample[0].shape)
                        
                    n = n + 1

trainloader = torch.utils.data.DataLoader(train_data, shuffle=True)
print("Loaded to trainloader...")
testloader = torch.utils.data.DataLoader(test_data, shuffle=True)
print("Loaded to testloader...")

model = models.densenet121(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.densenet121(pretrained=True)

for param in model.parameters():
    param.requires_grad = False
    
model.classifier = nn.Sequential(nn.Linear(1024, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(256, 11),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

model.to(device);
print(model)
epochs = 1
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
               
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
       
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()  
