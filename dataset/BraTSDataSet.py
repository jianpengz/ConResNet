import os.path as osp
import numpy as np
import random
from torch.utils import data
import nibabel as nib
from skimage.transform import resize


class BraTSDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(128, 160, 200), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_d, self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.is_mirror = mirror
        self.img_ids = [i_id.strip().split() for i_id in open(self.root + self.list_path)]

        if not max_iters==None:
                self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        for item in self.img_ids:
            filepath = item[0] +'/'+ osp.splitext(osp.basename(item[0]))[0]
            flair_path = filepath + '_flair.nii.gz'
            t1_path = filepath + '_t1.nii.gz'
            t1ce_path = filepath + '_t1ce.nii.gz'
            t2_path = filepath + '_t2.nii.gz'
            label_path = filepath + '_seg.nii.gz'
            name = osp.splitext(osp.basename(filepath))[0]
            flair_file = osp.join(self.root, flair_path)
            t1_file = osp.join(self.root, t1_path)
            t1ce_file = osp.join(self.root, t1ce_path)
            t2_file = osp.join(self.root, t2_path)
            label_file = osp.join(self.root, label_path)
            self.files.append({
                "flair": flair_file,
                "t1": t1_file,
                "t1ce": t1ce_file,
                "t2": t2_file,
                "label": label_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)

    def id2trainId(self, label):
        shape = label.shape
        results_map = np.zeros((3, shape[0], shape[1], shape[2]))

        NCR_NET = (label == 1)
        ET = (label == 4)
        WT = (label >= 1)
        TC = np.logical_or(NCR_NET, ET)

        results_map[0,:,:,:] = np.where(ET, 1, 0)
        results_map[1, :, :, :] = np.where(WT, 1, 0)
        results_map[2, :, :, :] = np.where(TC, 1, 0)
        return results_map

    def truncate(self, MRI):
        Hist, _ = np.histogram(MRI, bins=int(MRI.max()))
        idexs = np.argwhere(Hist >= 50)
        idex_max = np.float32(idexs[-1, 0])
        MRI[np.where(MRI >= idex_max)] = idex_max
        sig = MRI[0, 0, 0]
        MRI = np.where(MRI != sig, MRI - np.mean(MRI[MRI != sig]), 0 * MRI)
        MRI = np.where(MRI != sig, MRI / np.std(MRI[MRI != sig] + 1e-7), 0 * MRI)
        return MRI

    def __getitem__(self, index):
        datafiles = self.files[index]
        flairNII = nib.load(datafiles["flair"])
        t1NII = nib.load(datafiles["t1"])
        t1ceNII = nib.load(datafiles["t1ce"])
        t2NII = nib.load(datafiles["t2"])
        labelNII = nib.load(datafiles["label"])
        flair = self.truncate(flairNII.get_data())
        t1 = self.truncate(t1NII.get_data())
        t1ce = self.truncate(t1ceNII.get_data())
        t2 = self.truncate(t2NII.get_data())
        image = np.array([flair, t1, t1ce, t2])  # 4x240x240x150
        label = labelNII.get_data()
        image = image.astype(np.float32)
        label = label.astype(np.float32)

        if self.scale:
            scaler = np.random.uniform(0.9, 1.1)
        else:
            scaler = 1
        scale_d = int(self.crop_d * scaler)
        scale_h = int(self.crop_h * scaler)
        scale_w = int(self.crop_w * scaler)

        img_h, img_w, img_d = label.shape
        d_off = random.randint(0, img_d - scale_d)
        h_off = random.randint(15, img_h-15 - scale_h)
        w_off = random.randint(10, img_w-10 - scale_w)

        image = image[:, h_off: h_off + scale_h, w_off: w_off + scale_w, d_off: d_off + scale_d]
        label = label[h_off: h_off + scale_h, w_off: w_off + scale_w, d_off: d_off + scale_d]

        label = self.id2trainId(label)

        image = image.transpose((0, 3, 1, 2))  # Channel x Depth x H x W
        label = label.transpose((0, 3, 1, 2))     # Depth x H x W

        if self.is_mirror:
            randi = np.random.rand(1)
            if randi <= 0.3:
                pass
            elif randi <= 0.4:
                image = image[:, :, :, ::-1]
                label = label[:, :, :, ::-1]
            elif randi <= 0.5:
                image = image[:, :, ::-1, :]
                label = label[:, :, ::-1, :]
            elif randi <= 0.6:
                image = image[:, ::-1, :, :]
                label = label[:, ::-1, :, :]
            elif randi <= 0.7:
                image = image[:, :, ::-1, ::-1]
                label = label[:, :, ::-1, ::-1]
            elif randi <= 0.8:
                image = image[:, ::-1, :, ::-1]
                label = label[:, ::-1, :, ::-1]
            elif randi <= 0.9:
                image = image[:, ::-1, ::-1, :]
                label = label[:, ::-1, ::-1, :]
            else:
                image = image[:, ::-1, ::-1, ::-1]
                label = label[:, ::-1, ::-1, ::-1]

        if self.scale:
            image = resize(image, (4, self.crop_d, self.crop_h, self.crop_w), order=1, mode='constant', cval=0, clip=True, preserve_range=True)
            label = resize(label, (3, self.crop_d, self.crop_h, self.crop_w), order=0, mode='edge', cval=0, clip=True, preserve_range=True)
        image = image.astype(np.float32)
        label = label.astype(np.float32)

        # image -> res
        image_copy = np.zeros((4, self.crop_d, self.crop_h, self.crop_w)).astype(np.float32)
        image_copy[:, 1:, :, :] = image[:, 0:self.crop_d - 1, :, :]
        image_res = image - image_copy
        image_res[:, 0, :, :] = 0
        image_res = np.abs(image_res)

        # label -> res
        label_copy = np.zeros((3, self.crop_d, self.crop_h, self.crop_w)).astype(np.float32)
        label_copy[:, 1:, :, :] = label[:, 0:self.crop_d - 1, :, :]
        label_res = label - label_copy
        label_res[np.where(label_res == 0)] = 0
        label_res[np.where(label_res != 0)] = 1

        return image.copy(), image_res.copy(), label.copy(), label_res.copy()

class BraTSValDataSet(data.Dataset):
    def __init__(self, root, list_path):
        self.root = root
        self.list_path = list_path
        self.img_ids = [i_id.strip().split() for i_id in open(self.root + self.list_path)]
        self.files = []
        for item in self.img_ids:
            filepath = item[0] +'/'+ osp.splitext(osp.basename(item[0]))[0]
            flair_path = filepath + '_flair.nii.gz'
            t1_path = filepath + '_t1.nii.gz'
            t1ce_path = filepath + '_t1ce.nii.gz'
            t2_path = filepath + '_t2.nii.gz'
            label_path = filepath + '_seg.nii.gz'
            name = osp.splitext(osp.basename(filepath))[0]
            flair_file = osp.join(self.root, flair_path)
            t1_file = osp.join(self.root, t1_path)
            t1ce_file = osp.join(self.root, t1ce_path)
            t2_file = osp.join(self.root, t2_path)
            label_file = osp.join(self.root, label_path)
            self.files.append({
                "flair": flair_file,
                "t1": t1_file,
                "t1ce": t1ce_file,
                "t2": t2_file,
                "label": label_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)

    def id2trainId(self, label):
        shape = label.shape
        results_map = np.zeros((3, shape[0], shape[1], shape[2]))

        NCR_NET = (label == 1)
        ET = (label == 4)
        WT = (label >= 1)
        TC = np.logical_or(NCR_NET, ET)

        results_map[0, :, :, :] = np.where(ET, 1, 0)
        results_map[1, :, :, :] = np.where(WT, 1, 0)
        results_map[2, :, :, :] = np.where(TC, 1, 0)
        return results_map


    def truncate(self, MRI):
        Hist, _ = np.histogram(MRI, bins=int(MRI.max()))
        idexs = np.argwhere(Hist >= 50)
        idex_max = np.float32(idexs[-1, 0])
        MRI[np.where(MRI >= idex_max)] = idex_max
        sig = MRI[0, 0, 0]
        MRI = np.where(MRI != sig, MRI - np.mean(MRI[MRI != sig]), 0 * MRI)
        MRI = np.where(MRI != sig, MRI / np.std(MRI[MRI != sig] + 1e-7), 0 * MRI)
        return MRI

    def __getitem__(self, index):
        datafiles = self.files[index]

        flairNII = nib.load(datafiles["flair"])
        t1NII = nib.load(datafiles["t1"])
        t1ceNII = nib.load(datafiles["t1ce"])
        t2NII = nib.load(datafiles["t2"])
        labelNII = nib.load(datafiles["label"])

        flair = self.truncate(flairNII.get_data())
        t1 = self.truncate(t1NII.get_data())
        t1ce = self.truncate(t1ceNII.get_data())
        t2 = self.truncate(t2NII.get_data())
        image = np.array([flair, t1, t1ce, t2])  # 4x240x240x150
        label = labelNII.get_data()
        name = datafiles["name"]

        label = self.id2trainId(label)

        image = image.transpose((0, 3, 1, 2))  # Channel x Depth x H x W
        label = label.transpose((0, 3, 1, 2))     # Depth x H x W
        image = image.astype(np.float32)
        label = label.astype(np.float32)

        size = image.shape[1:]
        affine = labelNII.affine

        # image -> res
        cha, dep, hei, wei = image.shape
        image_copy = np.zeros((cha, dep, hei, wei)).astype(np.float32)
        image_copy[:, 1:, :, :] = image[:, 0:dep - 1, :, :]
        image_res = image - image_copy
        image_res[:, 0, :, :] = 0
        image_res = np.abs(image_res)

        return image.copy(), image_res.copy(), label.copy(), np.array(size), name, affine
