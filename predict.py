import json
import numpy as np
from PIL import Image
import matplotlib as plt
import argparse

import torch
from torchvision import models


#Inference Configuration

parser = argparse.ArgumentParser(description='Train image classifier')

parser.add_argument("-i", '--image_path', help = 'Image path', type = str, required=True)

parser.add_argument("-c", '--chkpnt', help = 'checkpoint path',
                    default = "image_classifier_6.pth", type = str)

parser.add_argument("-t", '--top_k', help = 'top k',
                    default = 5, type = int)

parser.add_argument("-v", '--device', help = 'device to use (cpu | cuda | cuda:n)',
                    default = "cuda", type = str)

parser.add_argument("-a", '--arch', help = 'featrue extraction arch. (pass name of function from torchvision.models)',
                    default = "densenet121", type = str)

parser.add_argument("-g", '--category_names_path', help = 'category name json path',
                    default = "cat_to_name.json", type = str)

cfg = parser.parse_args()

#############
#############


def load_model_state(chkpnt_pth, arch):
    model_chkpnt = torch.load(chkpnt_pth)
    m = models.__dict__[arch](pretrained=True)
    m.classifier = model_chkpnt['classifier']
    m.load_state_dict(model_chkpnt['state_dict'])
    m.class_to_idx = model_chkpnt['class_to_idx']
    
    for p in m.parameters():
        p.requires_grad = False
    
    return m


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406])

    width, height = image.size
    
    if width >= height:
        image.thumbnail((width, 256))
    elif height > width:
        image.thumbnail((256, height))
    w, h = image.size
    crop_w, crop_h = int((w - 224)//2), int((h - 224)//2)
    image = np.array(image)[crop_h:h-crop_h, crop_w:w-crop_w, :]

    image = image.astype(np.uint8)
    image = image / 255.
    image = (image - mean) / std
    image = image.transpose((2,0,1))
#     image = torch.from_numpy(np.array(image)).permute((2,0,1))
    return image


def imshow_input(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.permute((1, 2, 0)).cpu().numpy()
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path, model_, cfg):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    im = process_image(Image.open(image_path))
    im = torch.from_numpy(im).unsqueeze(0).float()
    im = im.to(cfg.device)
    
    with torch.no_grad():
        logps = model_.forward(im)

    logps = torch.exp(logps)
    top_five = logps.topk(cfg.top_k, dim=1)
    top_ps, calsses_ixs = top_five
    
    print('top_ps = ', top_ps)
    
    top_ps = top_ps.cpu().numpy()[0]
    classes = calsses_ixs.cpu().numpy().flatten()
    classes = [model_.class_to_idx[str(c)] for c in classes]
    classes = [cfg.cat_to_name[str(c)] for c in classes]    
    
    return top_ps, classes, im


def main(cfg):
    cat_to_name = None
    with open(cfg.category_names_path, 'r') as f:
        cat_to_name = json.load(f)
        
    cfg.cat_to_name = cat_to_name

    model = load_model_state(cfg.chkpnt, cfg.arch)
    model = model.to(device=cfg.device)
    model = model.eval()

    #image_path = 'flowers/test/2/image_05100.jpg'
    probs, classes, input_im = predict(cfg.image_path, model, cfg)

    print('confidence=', probs)
    print('predictions=', classes)


if __name__ == '__main__':
    main(cfg)

    
    