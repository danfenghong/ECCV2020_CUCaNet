from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os

# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, sp_matrix, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        input_image = input_image.data
        image_numpy = input_image[0].cpu().float().numpy()
    else:
        image_numpy = input_image

    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    elif image_numpy.shape[0] == sp_matrix.shape[0]:
        image_numpy = convert2rgb(image_numpy, sp_matrix)
    elif image_numpy.shape[0] > 3 and not image_numpy.shape[0] == sp_matrix.shape[0]:
        image_numpy = image_numpy[:3,:,:]
        image_numpy = image_numpy[::-1,:,:]
    elif image_numpy.shape[0] == 3:
        image_numpy = image_numpy[::-1,:,:]
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) ) * 255.0
    return image_numpy.astype(imtype)

def convert2rgb(image, sp_matrix):

    hsi_c, hsi_w, hsi_h = image.shape

    max_index = np.where(sp_matrix == np.max(sp_matrix, axis=0))

    assert(hsi_c == sp_matrix.shape[0])

    img = np.concatenate((image[max_index[0][2],:,:].reshape(-1,hsi_w, hsi_h),
                          image[max_index[0][1],:,:].reshape(-1,hsi_w, hsi_h),
                          image[max_index[0][0],:,:].reshape(-1,hsi_w, hsi_h)),axis=0)
    return img


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
