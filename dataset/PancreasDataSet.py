import numpy as np
import random
import torch
from torch.utils import data
from skimage.transform import resize

class PancreasDataSet(data.Dataset):
    def __init__(self, list_path, max_iters=None, crop_size=(64, 120, 120), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.list_path = list_path
        self.crop_d, self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip().split() for i_id in open(list_path)]
        if not max_iters==None:
                self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        for item in self.img_ids:
            filepath = item[0][0:-4] + 'images' + '/' + item[0][-4:]
            label_path = item[0][0:-4] + 'labels' + '/' + item[0][-4:]
            name = item[0][-4:]

            self.files.append({
                "img": filepath,
                "label": label_path,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)

    def id2trainId(self, label):
        shape = label.shape
        results_map = np.zeros((2, shape[0], shape[1], shape[2]))

        pancreas = (label==1)
        background = np.logical_not(pancreas)

        results_map[0,:,:,:] = np.where(background, 1, 0)
        results_map[1, :, :, :] = np.where(pancreas, 1, 0)
        return results_map

    def pre_precessing(self, image):
        image[image <= -100] = -100
        image[image >= 240] = 240
        image += 100
        image = image / 340
        return image

    def __getitem__(self, index):
        datafiles = self.files[index]
        # read nii file
        image = np.load(datafiles["img"] + '.npy')
        label = np.load(datafiles["label"] + '.npy')
        size = image.shape
        name = datafiles["name"]

        axes_index = np.argwhere(label == 1)
        one, two, three = axes_index[:, 0], axes_index[:, 1], axes_index[:, 2]
        min_x = np.min(one)
        max_x = np.max(one)
        min_x = min_x if min_x < 40 else min_x - 40
        max_x = size[0] if max_x >= size[0] - 40 - 1 else max_x + 40 + 1

        min_y = np.min(two)
        max_y = np.max(two)
        min_y = min_y if min_y < 40 else min_y - 40
        max_y = size[1] if max_y >= size[1] - 40 - 1 else max_y + 40 + 1

        min_z = np.min(three)
        max_z = np.max(three)
        min_z = min_z if min_z < 40 else min_z - 40
        max_z = size[2] if max_z >= size[2] - 40 - 1 else max_z + 40 + 1

        image = image[min_x:max_x, min_y:max_y, min_z:max_z]
        label = label[min_x:max_x, min_y:max_y, min_z:max_z]
        image = self.pre_precessing(image)

        if self.scale:
            scaler = np.random.uniform(0.9, 1.1)
        else:
            scaler = 1

        scale_d = int(self.crop_d * scaler)
        scale_h = int(self.crop_h * scaler)
        scale_w = int(self.crop_w * scaler)

        img_h, img_w, img_d = label.shape
        d_off = random.randint(0, img_d - scale_d)
        h_off = random.randint(0, img_h - scale_h)
        w_off = random.randint(0, img_w - scale_w)

        image = image[h_off: h_off + scale_h, w_off: w_off + scale_w, d_off: d_off + scale_d]
        label = label[h_off: h_off + scale_h, w_off: w_off + scale_w, d_off: d_off + scale_d]

        image = image.transpose((2, 0, 1))  
        label = label.transpose((2, 0, 1))  


        if self.is_mirror:
            randi = np.random.rand(1)
            if randi <= 0.3:
                pass
            elif randi <= 0.4:
                image = image[:, :, ::-1]
                label = label[:, :, ::-1]
            elif randi <= 0.5:
                image = image[:, ::-1, :]
                label = label[:, ::-1, :]
            elif randi <= 0.6:
                image = image[::-1, :, :]
                label = label[::-1, :, :]
            elif randi <= 0.7:
                image = image[:, ::-1, ::-1]
                label = label[:, ::-1, ::-1]
            elif randi <= 0.8:
                image = image[::-1, :, ::-1]
                label = label[::-1, :, ::-1]
            elif randi <= 0.9:
                image = image[::-1, ::-1, :]
                label = label[::-1, ::-1, :]
            else:
                image = image[::-1, ::-1, ::-1]
                label = label[::-1, ::-1, ::-1]

        if self.scale:
            image = resize(image, (self.crop_d, self.crop_h, self.crop_w), order=1, mode='constant', cval=0, clip=True, preserve_range=True)
            label = resize(label, (self.crop_d, self.crop_h, self.crop_w), order=0, mode='edge', cval=0, clip=True, preserve_range=True)

        image = np.array([image])
        label = np.array([label])

        image = image.astype(np.float32)
        label = label.astype(np.float32)

        # image -> res
        image_copy = np.zeros((1, self.crop_d, self.crop_h, self.crop_w)).astype(np.float32)
        image_copy[:, 1:, :, :] = image[:, 0:self.crop_d - 1, :, :]
        image_res = image - image_copy
        image_res[:, 0, :, :] = 0
        image_res = np.abs(image_res)

        # label -> res
        label_copy = np.zeros((1, self.crop_d, self.crop_h, self.crop_w)).astype(np.float32)
        label_copy[:, 1:, :, :] = label[:, 0:self.crop_d - 1, :, :]
        label_res = label - label_copy
        label_res[np.where(label_res == 0)] = 0
        label_res[np.where(label_res != 0)] = 1

        return image.copy(), image_res.copy(), label.copy(), label_res.copy(), np.array(size), name

class PancreasValDataSet(data.Dataset):
    def __init__(self, list_path):
        self.list_path = list_path
        self.img_ids = [i_id.strip().split() for i_id in open(list_path)]
        self.files = []
        for item in self.img_ids:
            filepath = item[0][0:-4] +'images' + '/' + item[0][-4:]
            label_path = item[0][0:-4] +'labels' + '/' + item[0][-4:]
            name = item[0][-4:]

            self.files.append({
                "img": filepath,
                "label": label_path,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)

    def id2trainId(self, label):
        shape = label.shape
        results_map = np.zeros((2, shape[0], shape[1], shape[2]))

        pancreas = (label == 1)
        background = np.logical_not(pancreas)

        results_map[0, :, :, :] = np.where(background, 1, 0)
        results_map[1, :, :, :] = np.where(pancreas, 1, 0)
        return results_map

    def pre_precessing(self, image):
        image[image <= -100] = -100
        image[image >= 240] = 240
        image += 100
        image = image / 340
        return image

    def __getitem__(self, index):
        datafiles = self.files[index]
        img = np.load(datafiles["img"] + '.npy')
        label = np.load(datafiles["label"] + '.npy')
        image = np.array([img]) 
        size = image.shape
        name = datafiles["name"]

        image = image.astype(np.float32)
        label = label.astype(np.float32)
        image[0, :, :, :] = self.pre_precessing(image[0, :, :, :])

        label = np.array([label])

        image = image.transpose((0, 3, 1, 2)) 
        label = label.transpose((0, 3, 1, 2)) 
        image = image.astype(np.float32)
        label = label.astype(np.float32)

        size = image.shape[1:]

        # image -> res
        cha, dep, hei, wei = image.shape
        image_copy = np.zeros((cha, dep, hei, wei)).astype(np.float32)
        image_copy[:, 1:, :, :] = image[:, 0:dep - 1, :, :]
        image_res = image - image_copy
        image_res[:, 0, :, :] = 0
        image_res = np.abs(image_res)

        return image.copy(), image_res.copy(), label.copy(), np.array(size), name

