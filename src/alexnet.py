
import numpy as np
import torch 
from torch import nn
from torch.hub import load_state_dict_from_url
from torchvision import transforms
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super().__init__()
        # change stride from 1 to 4 by Farah
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.features = []

        self.mean_img = [0.485, 0.456, 0.406]
        self.std_img =[0.229, 0.224, 0.225]
        if pretrained:
            model_urls = {
        'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
            }
            state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                                        progress=True)
            model_dict = self.state_dict()
            pretrained_keys = list(state_dict.keys())
            model_keys = list(model_dict.keys())
            for i in range(len(pretrained_keys)):
                model_dict[model_keys[i]].copy_(state_dict[pretrained_keys[i]])

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        fv1 = F.max_pool2d(x, kernel_size=3, stride=2)
        
        x = self.conv2(fv1)
        x = F.relu(x, inplace=True)
        fv2 = F.max_pool2d(x, kernel_size=3, stride=2)
        
        x = self.conv3(fv2)
        fv3 = F.relu(x, inplace=True)
        
        x = self.conv4(fv3)
        fv4 = F.relu(x, inplace=True)
        
        x = self.conv5(fv4)
        x = F.relu(x, inplace=True)
        fv5 = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.avgpool(fv5)
        out = torch.flatten(x, 1)
        x = self.classifier(out)
        self.features = [fv1, fv2, fv3, fv4, fv5, out]
        #return x, fv1, fv2, fv3, fv4, fv5
        return fv1, fv2, fv3, fv4, fv5, x
        # return x

    def responses(self, images, layer=1, batch_size=8, device=torch.device('cuda')):
        self.eval()
        nimg = images.shape[0]
        n_batches = int(np.ceil(nimg/batch_size))
        for k in range(n_batches):
            inds = np.arange(k*batch_size, min(nimg, (k+1)*batch_size))
            imgs = np.tile(images[inds][:,np.newaxis], (1,3,1,1)) / 255.
            imgs -= np.array(self.mean_img)[:,np.newaxis,np.newaxis]
            imgs /= np.array(self.std_img)[:,np.newaxis,np.newaxis]
            data = torch.from_numpy(imgs).to(device)
            output = self.forward(data)
            acts = output[layer].detach().cpu().numpy()
            if k==0:
                activations = np.zeros((nimg, *acts.shape[1:]))
            activations[inds] = acts
            
        return activations
    