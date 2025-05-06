import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import data.CI_torch_v2 as CI

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class ImgDataset(Dataset):
    def __init__(self, filelist, transform=None, transform_list=None,
                ehtarray='./data/EHT2017.txt', subarray=None,
                 date='2017-04-05', ra=187.7059167, dec=12.3911222, bw_hz=[230e9],
                 tint_sec=10, tadv_sec=48*60, tstart_hr=4.75, tstop_hr=6.5, psize=7.757018897750619e-12,
                 uvfits_files=None, ehtimAvg=False, avg_timescale=0,
                 ci_mask=None, ttype=None):
        self.filelist = filelist
        self.transform = transform
        if transform_list is None:
            self.transform_list = [transform] * len(self.filelist)
        else:
            self.transform_list = transform_list
        self.imgs = [np.load(f) for f in self.filelist]
        print([len(i) for i in self.imgs])
        self.closure = CI.Closure_Invariants(ehtarray=ehtarray, subarray=subarray,
                                             date=date, ra=ra, dec=dec, bw_hz=bw_hz, psize=psize,
                                             tint_sec=tint_sec, tadv_sec=tadv_sec, tstart_hr=tstart_hr, tstop_hr=tstop_hr,
                                             uvfits_files=uvfits_files, ehtimAvg=ehtimAvg, avg_timescale=avg_timescale,
                                             ci_mask=ci_mask, ttype=ttype)
        
        
        # self.imgs = np.concatenate(self.imgs)

        self.class_label_names = [f.split('_')[-1].split('.')[0] for f in self.filelist]
        print(self.class_label_names)

        self.class_labels = [np.zeros((len(i), len(self.class_label_names))) for i in self.imgs]

        for i in range(len(self.imgs)):
            self.class_labels[i][:, self.class_label_names.index(self.class_label_names[i])] = 1

        self.imgs = np.concatenate(self.imgs)
        self.class_labels = (np.concatenate(self.class_labels))

        # normalise every image
        for i in range(len(self.imgs)):
            # replace NaNs
            self.imgs[i] = np.nan_to_num(self.imgs[i])
            self.imgs[i] = (self.imgs[i] - np.nanmin(self.imgs[i])) / (np.nanmax(self.imgs[i]) - np.nanmin(self.imgs[i]))
            self.imgs[i] = self.imgs[i].swapaxes(-1, -2) # swap axes
            # self.imgs[i] = self.imgs[i]/np.nansum(self.imgs[i]) # normalise to sum to 1


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.imgs[idx]
        class_label = self.class_labels[idx]

        if self.transform or self.transform_list is not None:
            transform = self.transform_list[np.where(class_label == 1)[0][0]]
            image = transform(image)

        # ci = self.closure.FTCI(np.array([image])).reshape(-1)
        ci = np.array([0])

        image = image.to(dtype=torch.float32)
        class_label = torch.from_numpy(class_label).long()
        ci = torch.from_numpy(ci).float()

        return image, class_label
