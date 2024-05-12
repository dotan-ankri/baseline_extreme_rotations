####from doppelgangers
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

def get_resized_wh(w, h, resize=None):
    if resize is not None:  # resize the longer edge
        scale = resize / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    else:
        w_new, h_new = w, h
    return w_new, h_new

def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    if w_new == 0:
        w_new = df
    if h_new == 0:
        h_new = df
    return w_new, h_new

def get_resized_and_pad(img_raw, img_size=256, df=8, padding=True):
    w, h = img_raw.squeeze(0).shape[1], img_raw.squeeze(0).shape[0]
    w_new, h_new = get_resized_wh(w, h, img_size)
    w_new, h_new = get_divisible_wh(w_new, h_new, df)

    if padding:  # padding
        pad_to = max(h_new, w_new)    
        mask = np.zeros((1,pad_to, pad_to), dtype=bool)
        mask[:,:h_new,:w_new] = True
        mask = mask[:,::8,::8]
    transform = T.Resize((h_new,w_new),antialias=True)
    image = transform(img_raw)
    pad_image = np.zeros((1,1, pad_to, pad_to), dtype=np.float32)
    pad_image[0,0,:h_new,:w_new]=image/255.

    return pad_image.squeeze(0), mask.squeeze(0)

def get_crop_square_and_resize(img_raw, img_size=256, df=8, padding=True):
    
    if not torch.is_tensor(img_raw):
        img_raw = T.ToTensor()(img_raw)
      
    w, h = img_raw.squeeze(0).shape[1], img_raw.squeeze(0).shape[0]
    new_size = min(w,h)
    transform = T.Compose([T.CenterCrop((new_size,new_size)),T.Resize((img_size,img_size),antialias=True)])
    image = transform(img_raw)
    
    return image
