import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader,random_split
import math
#import matplotlib.pyplot as plt
from torch import optim  
from torch import nn 
import torch.nn.functional as F 
import cv2


class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,4,3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout=nn.Dropout()
        self.conv2 = nn.Conv2d(4,8,3,stride=1,padding=1 )
        self.fc1 = nn.Linear(8 * 7 * 7, 150)
        self.fc2 = nn.Linear(150, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        #x = torch.flatten(x, 1)
        x=x.view(-1,8*7*7)
        x=self.dropout(x)
        x = F.relu(self.fc1(x))
        x=self.fc2(x)

        return x

in_channels=1
num_classes=20
#loading model for evalution 
FILE=r"model.pth"
loaded_model = CNN(in_channels, num_classes)
loaded_model.load_state_dict(torch.load(FILE)) # it takes the loaded dictionary, not the path file itself
loaded_model.eval()
classes=('airplane','ant','banana','baseball','bird','bucket','butterfly','cat','coffee cup','dolphin','donut','duck','fish','leaf','mountain','pencil','smiley face','snake','umbrella','wine bottle')

print("1.Press Spacebar to clear pad\n2.To exit press 'q'\n")


drawing = False # true if mouse is pressed
pt1_x , pt1_y = None , None
l=0
# mouse callback function
def line_drawing(event,x,y,flags,param):
    global pt1_x,pt1_y,drawing,l

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        pt1_x,pt1_y=x,y
        
    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=12)
            pt1_x,pt1_y=x,y
            
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=12)        
        l=27

img = np.zeros((600,600,3), np.uint8)
cv2.namedWindow('test draw')
cv2.setMouseCallback('test draw',line_drawing)

while(1):
    cv2.imshow('test draw',img)
    p=cv2.waitKey(1) & 0xFF
   
    if p==ord('q'):
        break
    if p==32:
        img = np.zeros((600,600,3), np.uint8)
    if l==27:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #edge ddetct and crop 
        edged = cv2.Canny(gray, 0, 250) 
        (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        x,y,w,h = cv2.boundingRect(cnts[-1]) 
        new_img=gray[y:y+h,x:x+w] 
        #cv2.imwrite('imgg.png', new_img)
         
        input_img=cv2.resize(new_img,(28,28),interpolation=cv2.INTER_AREA)
        x=torch.from_numpy(input_img)
        x=x.reshape(1,1,28,28)

        res=loaded_model(x.float())
        i=torch.argmax(res)

        print("it's a ",classes[i.item()])
        l=0
    
    
cv2.destroyAllWindows()