import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from videoDataset import videoDataset

import os

import time

import cv2
from os.path import isfile,join

VideoLoad = videoDataset()

samples=[]

n = 0

for filename in os.listdir("D:/UCF11/data"):
	for filename2 in os.listdir("D:/UCF11/data/"+filename):
		for filename3 in os.listdir("D:/UCF11/data/"+filename+'/'+filename2):
			if(filename3.split(".")[0].split("_")[-1] == 'resized'):
					sample = videoDataset.readVideo(VideoLoad, "D:/UCF11/data/"+filename+'/'+filename2+'/'+filename3)
					n = n + 1
					if(n%3==0):
						testloader = torch.utils.data.DataLoader(sample[0])
						print("Loaded to testloader!")
						
					else:
						trainloader = torch.utils.data.DataLoader(sample[0])
						print("Loaded to trainloader!")
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