'''
# @aribra
'''

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from torch import optim

import os
import numpy as np
from pylab import *
from PIL import Image

import argparse
from collections import OrderedDict



#Training Configuration

parser = argparse.ArgumentParser(description='Train image classifier')

parser.add_argument("-d", '--data_directory', help = 'data directory',
                    default = "flowers", type = str)

parser.add_argument("-s", '--save_dir', help = 'save chkpnts here',
                    default = "./", type = str)

parser.add_argument("-a", '--arch', help = 'featrue extraction arch. (pass name of function from torchvision.models)',
                    default = "densenet121", type = str)

parser.add_argument("-b", '--batch_size', help = 'batch size',
                    default = 64, type = int)

parser.add_argument("-l", '--lr', help = 'leraning rate',
                    default = 3e-3, type = float)

parser.add_argument("-e", '--nb_epochs', help = 'number of epochs',
                    default = 7, type = int)

parser.add_argument("-o", '--base_network_out_features', help = 'base_network_out_features',
                    default = 1024, type = int)

parser.add_argument("-n", '--nb_classes', help = 'number of output classes',
                    default = 102, type = int)

parser.add_argument("-v", '--device', help = 'device to use (cpu | cuda | cuda:n)',
                    default = "cuda", type = str)

cfg = parser.parse_args()


#############
#############


def load_data(data_directory, batch_size=64):

    train_dir = data_directory + '/train'
    valid_dir = data_directory + '/valid'
    test_dir = data_directory + '/test'


    imagenet_std = (0.229, 0.224, 0.225)
    imagenet_mean = (0.485, 0.456, 0.406)
    input_size = 224


    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(50),
                                           transforms.RandomResizedCrop(input_size),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
                                            ]) 

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(input_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
                                          ])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
    validloader = torch.utils.data.DataLoader(valid_data, shuffle=True, batch_size=batch_size) 
    testloader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=batch_size)
    
    
    data_folders = (train_data, valid_data, test_data)
    data_loaders = (trainloader, validloader, testloader)
    
    return data_folders, data_loaders


def get_classifier_layers(out_features, nb_classes):
    layers = []
    
    downsample_palette = [out_features, 1024, 512, 256, nb_classes]
    start_ix = out_features - np.array(downsample_palette)
    start_ix = start_ix[start_ix < 0]
    start_ix = 0 if len(start_ix)==0 else  (np.flatnonzero(start_ix)) [-1]
    nb_layers = len(downsample_palette) - start_ix - 1

    for i in range(nb_layers):
        if i == nb_layers-1+start_ix:
            break

        nb_in, nb_out = downsample_palette[start_ix+i], downsample_palette[start_ix+i+1]        
        fc = ('fc_'+str(i), nn.Linear(nb_in, nb_out))
        relu = ('relu', nn.ReLU())
        dropout = ('dropout_'+str(i), nn.Dropout(p=0.15))
        
        layers.append (fc)
        layers.append (relu)
        layers.append (dropout)

    output_layer = ('output', nn.LogSoftmax(dim=1))
    layers.append ( output_layer )

    return layers


def init_model(cfg):
    model = models.__dict__[cfg.arch](pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    layers = get_classifier_layers(cfg.base_network_out_features, cfg.nb_classes)
    classifier = nn.Sequential( 
        OrderedDict( layers ) 
    )
    model.classifier = classifier

    return model


def save_chkpnt(model, epoch, cfg):
    model_state = {
        'input': 224,
        'output': cfg.nb_classes,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }
       
    chkpnt_path = os.path.join( cfg.save_dir, 'image_classifier_{}.pth' . format(epoch) )
    torch.save(model_state, chkpnt_path)

    
def train(model, trainloader, validloader, cfg):
    device = 'cuda' if torch.cuda.is_available() else cfg.device
    
    model = model.to(cfg.device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(params=model.classifier.parameters(), lr=cfg.lr)

    
    train_loss = 0
    val_every = 15
    steps = 0
    
    for epoch in range(cfg.nb_epochs):
        print('Start training epoch {}...'.format(epoch+1))
        
        model.train()
        for images, labels in trainloader:
            steps += 1

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(images)
            loss = criterion(logps, labels)
    #         loss.requires_grad = True
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if steps % val_every == 0:
                val_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():                
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)

                        logps = model.forward(images)
                        loss = criterion(logps, labels)
                        val_loss += loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()               

                print(f"Epoch {epoch+1}/{cfg.nb_epochs}.. "
                      f"Train loss: {train_loss/val_every:.3f}.. "
                      f"Test loss: {val_loss/len(validloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(validloader):.3f}")
                
                save_chkpnt(model, epoch+1, cfg)
                
                train_loss = 0
                model.train()

    return model


def main():
    #Load data
    data_folders, data_loaders = load_data(cfg.data_directory, batch_size=cfg.batch_size)
    train_data, valid_data, test_data = data_folders
    trainloader, validloader, testloader = data_loaders

    #Init network
    model = init_model(cfg)
    model.class_to_idx = train_data.class_to_idx

    #Train
    train(model, trainloader, validloader, cfg)


if __name__ == '__main__':
    main(cfg)

