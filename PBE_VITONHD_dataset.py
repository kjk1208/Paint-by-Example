#1
import os
from os.path import join as opj

import cv2
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
import torch

def imread(
        p, h, w, 
        is_mask=False, 
        in_inverse_mask=False, 
        img=None
):
    if img is None:
        img = cv2.imread(p)
    if not is_mask:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w,h))
        img = (img.astype(np.float32) / 127.5) - 1.0  # [-1, 1]
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (w,h))
        img = (img >= 128).astype(np.float32)  # 0 or 1
        img = img[:,:,None]
        if in_inverse_mask:
            img = 1-img
    return img

def imread_for_albu(
        p, 
        is_mask=False, 
        in_inverse_mask=False, 
        cloth_mask_check=False, 
        use_resize=False, 
        height=512, 
        width=384,
):
    img = cv2.imread(p)
    if use_resize:
        img = cv2.resize(img, (width, height))
    if not is_mask:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = (img>=128).astype(np.float32)
        if cloth_mask_check:
            if img.sum() < 30720*4:
                img = np.ones_like(img).astype(np.float32)
        if in_inverse_mask:
            img = 1 - img
        img = np.uint8(img*255.0)
    return img
def norm_for_albu(img, is_mask=False):
    if not is_mask:
        img = (img.astype(np.float32)/127.5) - 1.0
    else:
        img = img.astype(np.float32) / 255.0
        img = img[:,:,None]
    return img

class VITONHDDataset_PBE(Dataset):
    def __init__(
            self, 
            data_root_dir, 
            img_H, 
            img_W, 
            is_paired=True, 
            is_test=False, 
            is_sorted=False, 
            transform_size=None, 
            transform_color=None,
            **kwargs
        ):
        self.drd = data_root_dir
        self.img_H = img_H
        self.img_W = img_W
        self.pair_key = "paired" if is_paired else "unpaired"
        self.data_type = "train" if not is_test else "test"
        self.is_test = is_test
        self.resize_ratio_H = 1.0
        self.resize_ratio_W = 1.0

        self.resize_transform = A.Resize(img_H, img_W)
        self.transform_size = None
        self.transform_crop_person = None
        self.transform_crop_cloth = None
        self.transform_color = None

        #### spatial aug >>>>
        ################################################
        transform_crop_person_lst = []
        transform_crop_cloth_lst = []
        transform_size_lst = [A.Resize(int(img_H*self.resize_ratio_H), int(img_W*self.resize_ratio_W))]
    
        if transform_size is not None:
            if "hflip" in transform_size:
                transform_size_lst.append(A.HorizontalFlip(p=0.5))
                print(f"transform_size_lst : hflip")

            if "shiftscale" in transform_size:
                transform_crop_person_lst.append(A.ShiftScaleRotate(rotate_limit=0, shift_limit=0.2, scale_limit=(-0.2, 0.2), border_mode=cv2.BORDER_CONSTANT, p=0.5, value=0))
                transform_crop_cloth_lst.append(A.ShiftScaleRotate(rotate_limit=0, shift_limit=0.2, scale_limit=(-0.2, 0.2), border_mode=cv2.BORDER_CONSTANT, p=0.5, value=0))
                print(f"transform_crop_person_lst : shiftscale")
                print(f"transform_crop_cloth_lst : shiftscale")
            else:
                print(f"no shiftscale3")

        self.transform_crop_person = A.Compose(
                transform_crop_person_lst,
                additional_targets={#"gt_cloth_warped":"image",
                                    "agn_mask":"image",                                     
                                    "gt_cloth_warped_agn":"image", 
                                    "image_densepose":"image", 
                                    }
        )
        self.transform_crop_cloth = A.Compose(
                transform_crop_cloth_lst,
                #additional_targets={"cloth":"image"}
        )
        
        self.transform_size = A.Compose(
                transform_size_lst,
                additional_targets={#"gt_cloth_warped":"image",
                                    "agn_mask":"image", 
                                    "cloth":"image", 
                                    "gt_cloth_warped_agn":"image", 
                                    "image_densepose":"image", 
                                    }
            )
        #### spatial aug <<<<

        #### color aug >>>>
        if transform_color is not None:
            transform_color_lst = []
            for t in transform_color:
                print(f"t : {t}")
                if t == "hsv":
                    transform_color_lst.append(A.HueSaturationValue(5,5,5,p=0.5))
                    print(f"transform_color_lst : hsv")
                elif t == "bright_contrast":
                    transform_color_lst.append(A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.02), contrast_limit=(-0.3, 0.3), p=0.5))
                    print(f"transform_color_lst : bright_contrast")

            self.transform_color = A.Compose(
                transform_color_lst,
                additional_targets={#"gt_cloth_warped":"image", 
                                    "cloth":"image",  
                                    "gt_cloth_warped_agn":"image",
                                    }
            )
        #### non-spatial aug <<<<
                    
        assert not (self.data_type == "train" and self.pair_key == "unpaired"), f"train must use paired dataset"
        
        im_names = []
        c_names = []
        with open(opj(self.drd, f"{self.data_type}_pairs.txt"), "r") as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)
        if is_sorted:
            im_names, c_names = zip(*sorted(zip(im_names, c_names)))
        self.im_names = im_names
        
        self.c_names = dict()
        self.c_names["paired"] = im_names
        self.c_names["unpaired"] = c_names

    def __len__(self):
        return len(self.im_names)
    
    def __getitem__(self, idx):
        img_fn = self.im_names[idx]
        cloth_fn = self.c_names[self.pair_key][idx]
        if self.transform_size is None and self.transform_color is None:
            # 1.batch['GT']
            gt_cloth_warped = imread(
                opj(self.drd, self.data_type, "gt_cloth_warped", self.im_names[idx]),
                self.img_H,
                self.img_W,                
            )                        
            # 2.batch['inpaint_mask']
            agn_mask = imread(
                opj(self.drd, self.data_type, "agnostic-mask", self.im_names[idx]), 
                self.img_H, 
                self.img_W, 
                is_mask=True, 
                in_inverse_mask=True
            )
            # 3.batch['ref_image']
            cloth = imread(
                opj(self.drd, self.data_type, "cloth", self.c_names[self.pair_key][idx]), 
                self.img_H, 
                self.img_W
            )
            # 4.batch['inpaint_image']
            gt_cloth_warped_agn = imread(
                opj(self.drd, self.data_type, "gt_cloth_warped+agn_mask", self.im_names[idx]),
                self.img_H,
                self.img_W,
            )            
            # 5.batch['image_densepose']
            image_densepose = imread(opj(self.drd, self.data_type, "image-densepose", self.im_names[idx]), self.img_H, self.img_W)

        else:            
            #print(opj(self.drd, self.data_type, "gt_cloth_warped", self.im_names[idx]))
            # 1.batch['GT']
            gt_cloth_warped = imread_for_albu(opj(self.drd, self.data_type, "gt_cloth_warped", self.im_names[idx]))
            # 2.batch['inpaint_mask']
            agn_mask = imread_for_albu(opj(self.drd, self.data_type, "agnostic-mask", self.im_names[idx]), is_mask=True)
            # 3.batch['ref_image']
            cloth = imread_for_albu(opj(self.drd, self.data_type, "cloth", self.c_names[self.pair_key][idx]))
            # 4.batch['inpaint_image']
            gt_cloth_warped_agn = imread_for_albu(opj(self.drd, self.data_type, "gt_cloth_warped+agn_mask", self.im_names[idx]))
            # 5.batch['image_densepose']
            image_densepose = imread_for_albu(opj(self.drd, self.data_type, "image-densepose", self.im_names[idx]))
            
            
            
            if self.transform_size is not None:
                transformed = self.transform_size(
                    image=gt_cloth_warped,                     
                    agn_mask=agn_mask, 
                    cloth=cloth, 
                    gt_cloth_warped_agn=gt_cloth_warped_agn, 
                    image_densepose=image_densepose,                    
                )
                gt_cloth_warped=transformed["image"]                
                agn_mask=transformed["agn_mask"]
                cloth=transformed["cloth"]
                gt_cloth_warped_agn=transformed["gt_cloth_warped_agn"]
                image_densepose=transformed["image_densepose"]
               
            if self.transform_crop_person is not None:
                transformed_image = self.transform_crop_person(
                    image=gt_cloth_warped,
                    agn_mask=agn_mask,
                    gt_cloth_warped_agn=gt_cloth_warped_agn,
                    image_densepose=image_densepose,
                )

                gt_cloth_warped=transformed_image["image"]                
                agn_mask=transformed_image["agn_mask"]                
                gt_cloth_warped_agn=transformed_image["gt_cloth_warped_agn"]
                image_densepose=transformed_image["image_densepose"]

            if self.transform_crop_cloth is not None:
                transformed_cloth = self.transform_crop_cloth(
                    image=cloth,                    
                )

                cloth=transformed_cloth["image"]

            agn_mask = 255 - agn_mask
            if self.transform_color is not None:
                transformed = self.transform_color(
                    image=gt_cloth_warped,                     
                    cloth=cloth,
                    gt_cloth_warped_agn=gt_cloth_warped_agn, 
                )

                gt_cloth_warped=transformed["image"]                
                cloth=transformed["cloth"]
                gt_cloth_warped_agn=transformed["gt_cloth_warped_agn"]
                
                gt_cloth_warped_agn = gt_cloth_warped_agn * agn_mask[:,:,None].astype(np.float32)/255.0 + 128 * (1 - agn_mask[:,:,None].astype(np.float32)/255.0)
                
            gt_cloth_warped = norm_for_albu(gt_cloth_warped)
            gt_cloth_warped_agn = norm_for_albu(gt_cloth_warped_agn)
            agn_mask = norm_for_albu(agn_mask, is_mask=True)
            cloth = norm_for_albu(cloth)
            image_densepose = norm_for_albu(image_densepose)
            
            #ToTensor
            gt_cloth_warped = torch.from_numpy(gt_cloth_warped).permute(2, 0, 1)
            gt_cloth_warped_agn = torch.from_numpy(gt_cloth_warped_agn).permute(2, 0, 1)
            agn_mask = torch.from_numpy(agn_mask).permute(2, 0, 1)
            cloth = torch.from_numpy(cloth).permute(2, 0, 1)
            image_densepose = torch.from_numpy(image_densepose).permute(2, 0, 1)
            #ToTensor

            
        return {
            "GT":gt_cloth_warped,
            "inpaint_image":gt_cloth_warped_agn,
            "inpaint_mask":agn_mask,
            "ref_imgs":cloth,
            "image_densepose":image_densepose,}
        # return dict(
        #     GT=gt_cloth_warped,            
        #     inpaint_image=gt_cloth_warped_agn,
        #     inpaint_mask=agn_mask,
        #     ref_imgs=cloth,
        #     image_densepose=image_densepose,
        #     txt="",
        #     img_fn=img_fn,
        #     cloth_fn=cloth_fn,
        # )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    from torch.utils.data import DataLoader, random_split
    import torch

    dataset = VITONHDDataset_PBE(
        data_root_dir="../stableviton_lightning/datasets/", 
        img_H=512,
        img_W=384,
        is_paired=True,
        is_test=False,
        is_sorted=False,
        transform_size=["hflip", "shiftscale"],
        transform_color=["hsv", "bright_contrast"]
    )

    subset_size = len(dataset) // 100
    subset_indices = np.random.choice(len(dataset), subset_size, replace=False)
    subset_dataset = torch.utils.data.Subset(dataset, subset_indices)

    data_loader = DataLoader(subset_dataset, batch_size=10, shuffle=False)

    for batch in data_loader:
        gt_cloth_warped = batch['GT']
        gt_cloth_warped_agn = batch['inpaint_image']
        agn_mask = batch['inpaint_mask']
        cloth = batch['ref_imgs']
        image_densepose = batch['image_densepose']
        
        agn_mask_3ch = agn_mask.repeat((1, 1, 1, 3))
        
        gt_cloth_warped = gt_cloth_warped.permute(0, 3, 1, 2)
        gt_cloth_warped_agn = gt_cloth_warped_agn.permute(0, 3, 1, 2)
        agn_mask_3ch = agn_mask_3ch.permute(0, 3, 1, 2)
        cloth = cloth.permute(0, 3, 1, 2)
        image_densepose = image_densepose.permute(0, 3, 1, 2)

        images = [gt_cloth_warped, gt_cloth_warped_agn, agn_mask_3ch, cloth, image_densepose]
        
        

        all_images = torch.cat(images, dim=0)
        print(f"all_images shape : {all_images.shape}")

        img_grid = make_grid(all_images, nrow=10)

        
        plt.figure(figsize=(15, 15))
        plt.imshow(np.transpose(img_grid.numpy(), (1, 2, 0)))
        plt.axis('off')
        plt.title('Sample Images from VITON HDDataset')
        break

    plt.savefig('VITON_data.png')