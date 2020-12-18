import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from frame_utils import writeFlow
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
import subprocess as sp

import re
import struct
import skimage
import skimage.io

import torch
from torch.utils.data import Dataset

class KLens(Dataset):
    def __init__(self, left,right, split="infer",):
        super(KLens, self).__init__()
        self.split = split
        file_list = {}
        file_list['infer'] = []
        #file_list = {}
        #file_list['train'] = [[os.path.join(root_path,"KLE_"+filenum+".jpg4.png"),os.path.join(root_path,"KLE_"+filenum+".jpg5.png")] for filenum in filenumberlist]
        #file_list['valid'] = []
        #file_list['test'] = [[os.path.join(root_path,"KLE_"+filenum+".jpg4.png"),os.path.join(root_path,"KLE_"+filenum+".jpg5.png")] for filenum in filenumberlist]
        #file_list['train+valid'] = [[os.path.join(root_path,"KLE_"+filenum+".jpg4.png"),os.path.join(root_path,"KLE_"+filenum+".jpg5.png")] for filenum in filenumberlist]
        for i,v in enumerate(left):
            if(os.path.isfile(v)):
                if(os.path.isfile(right[i])):
                    file_list["infer"].extend([[v,right[i]]])
                else:
                    print("Skipping ",v ," ",right[i],", File not found: ",right[i])
            else:
                print("Skipping ",v ," ",right[i],", File not found: ",v)
        """
        file_list["valid"].extend([[os.path.join(root_path,"KLE_0309_exp_sub5.jpg"),os.path.join(root_path,"KLE_0309_exp_sub6.jpg")],[os.path.join(root_path,"KLE_0730_sub5.jpg"),os.path.join(root_path,"KLE_0730_sub6.jpg")],[os.path.join(root_path,"KLE_0747_sub5.jpg"),os.path.join(root_path,"KLE_0747_sub6.jpg")],[os.path.join(root_path,"KLE_9797clean_sub5.jpg"),os.path.join(root_path,"KLE_9797clean_sub6.jpg")],[os.path.join(root_path,"KLE_9803clean_sub5.jpg"),os.path.join(root_path,"KLE_9803clean_sub6.jpg")],[os.path.join(root_path,"NKM_0063_sub5.jpg"),os.path.join(root_path,"NKM_0063_sub6.jpg")],[os.path.join(root_path,"NKM_0109_sub5.jpg"),os.path.join(root_path,"NKM_0109_sub6.jpg")],[os.path.join(root_path,"scene_1_sub5.jpg"),os.path.join(root_path,"scene_1_sub6.jpg")],[os.path.join(root_path,"DSC_0608_sub4.jpg"),os.path.join(root_path,"DSC_0608_sub5.jpg")]])
        file_list["test"].extend([[os.path.join(root_path,"KLE_0309_exp_sub5.jpg"),os.path.join(root_path,"KLE_0309_exp_sub6.jpg")],[os.path.join(root_path,"KLE_0730_sub5.jpg"),os.path.join(root_path,"KLE_0730_sub6.jpg")],[os.path.join(root_path,"KLE_0747_sub5.jpg"),os.path.join(root_path,"KLE_0747_sub6.jpg")],[os.path.join(root_path,"KLE_9797clean_sub5.jpg"),os.path.join(root_path,"KLE_9797clean_sub6.jpg")],[os.path.join(root_path,"KLE_9803clean_sub5.jpg"),os.path.join(root_path,"KLE_9803clean_sub6.jpg")],[os.path.join(root_path,"NKM_0063_sub5.jpg"),os.path.join(root_path,"NKM_0063_sub6.jpg")],[os.path.join(root_path,"NKM_0109_sub5.jpg"),os.path.join(root_path,"NKM_0109_sub6.jpg")],[os.path.join(root_path,"scene_1_sub5.jpg"),os.path.join(root_path,"scene_1_sub6.jpg")]])
        file_list["train+valid"].extend([[os.path.join(root_path,"KLE_0309_exp_sub5.jpg"),os.path.join(root_path,"KLE_0309_exp_sub6.jpg")],[os.path.join(root_path,"KLE_0730_sub5.jpg"),os.path.join(root_path,"KLE_0730_sub6.jpg")],[os.path.join(root_path,"KLE_0747_sub5.jpg"),os.path.join(root_path,"KLE_0747_sub6.jpg")],[os.path.join(root_path,"KLE_9797clean_sub5.jpg"),os.path.join(root_path,"KLE_9797clean_sub6.jpg")],[os.path.join(root_path,"KLE_9803clean_sub5.jpg"),os.path.join(root_path,"KLE_9803clean_sub6.jpg")],[os.path.join(root_path,"NKM_0063_sub5.jpg"),os.path.join(root_path,"NKM_0063_sub6.jpg")],[os.path.join(root_path,"NKM_0109_sub5.jpg"),os.path.join(root_path,"NKM_0109_sub6.jpg")],[os.path.join(root_path,"scene_1_sub5.jpg"),os.path.join(root_path,"scene_1_sub6.jpg")]])
        file_list["train"].extend([[os.path.join(root_path2,"9-AIT_pins_2.jpg"),os.path.join(root_path2,"9-AIT_pins_3.jpg")],[os.path.join(root_path2,"10-Hela_2.jpg"),os.path.join(root_path2,"10-Hela_3.jpg")],[os.path.join(root_path2,"11-Hela_1_2.jpg"),os.path.join(root_path2,"11-Hela_1_3.jpg")],])
        file_list["train"].extend([[os.path.join(root_path2,"9-AIT_pins_2.jpg"),os.path.join(root_path2,"9-AIT_pins_0.jpg")],[os.path.join(root_path2,"10-Hela_2.jpg"),os.path.join(root_path2,"10-Hela_0.jpg")],[os.path.join(root_path2,"11-Hela_1_2.jpg"),os.path.join(root_path2,"11-Hela_1_0.jpg")],])
        file_list["train"].extend([[os.path.join(root_path2,"9-AIT_pins_2.jpg"),os.path.join(root_path2,"9-AIT_pins_1.jpg")],[os.path.join(root_path2,"10-Hela_2.jpg"),os.path.join(root_path2,"10-Hela_1.jpg")],[os.path.join(root_path2,"11-Hela_1_2.jpg"),os.path.join(root_path2,"11-Hela_1_1.jpg")],])
        file_list["train"].extend([[os.path.join(root_path2,"9-AIT_pins_2.jpg"),os.path.join(root_path2,"9-AIT_pins_4.jpg")],[os.path.join(root_path2,"10-Hela_2.jpg"),os.path.join(root_path2,"10-Hela_4.jpg")],[os.path.join(root_path2,"11-Hela_1_2.jpg"),os.path.join(root_path2,"11-Hela_1_4.jpg")],])
        """
        self.dataset = file_list
        
        
    def __len__(self):
        return len(self.dataset[self.split])

    def __getitem__(self, idx):
        im0_path, im1_path = self.dataset[self.split][idx]

        images= self.load_image_list([im0_path,im1_path])
        # print(images[])
        for i in range(images.shape[0]-1):
            image0 = images[i,None]
            image1 = images[i+1,None]
        # img0 = skimage.io.imread(im0_path)
        # img1 = skimage.io.imread(im1_path)
        # img0 = torch.tensor(img0/255.).float()
        # img1 = torch.tensor(img1/255.).float()

        return image0, image1, im0_path , im1_path
    
    def load_image(self,imfile):
        img = np.array(Image.open(imfile)).astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img
    
    def load_image_list(self,image_files):
        images = []
        for imfile in image_files:
            images.append(self.load_image(imfile))
    
        images = torch.stack(images, dim=0)
        # images = images.to("cuda")

        padder = InputPadder(images.shape)
        return padder.pad(images)[0]

def get_gpu_memory():
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values



def disparity_assessment(ref_image, neighbor_image, disp,path_ref,path_meas,out_dir):
    # Warping
    root_path = out_dir
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    writeFlow(
        os.path.join(
            root_path,
            os.path.basename(
                os.path.splitext(path_ref)[0]
            )+
            "_"+
            os.path.basename(
                os.path.splitext(path_meas)[0]
            )+
            ".flo"),
        disp
    )
    cv2.imwrite(
        os.path.join(
            root_path,
            os.path.basename(
                os.path.splitext(path_ref)[0]
            )+
            "_"+
            os.path.basename(
                os.path.splitext(path_meas)[0]
            )+
            ".jpg"
        ),
        flow_viz.flow_to_image(disp)[:,:,[2,1,0]]
    )
    print("Output written to "+out_dir)


DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img


def load_image_list(image_files):
    images = []
    for imfile in sorted(image_files):
        images.append(load_image(imfile))

    images = torch.stack(images, dim=0)
    images = images.to(DEVICE)

    padder = InputPadder(images.shape)
    return padder.pad(images)[0]

def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    #print(flo.cpu().numpy().shape)
    flo = flo[0].permute(1,2,0).cpu().numpy()
    #print(flo.shape)
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    cv2.imwrite('image0.jpg', flo[:,:,])

def demo(args,data_loader):
    # print("Used memory:",get_gpu_memory()[0])

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    #print("Used memory:",get_gpu_memory()[0])
    print("starting inference")

    for idx, sample in enumerate(data_loader):
        with torch.no_grad():
            im0, im1, path_ref,path_meas = sample
            print(args.model,path_ref,path_meas)
            # images = load_image_list(imagenames)
            # for i in range(images.shape[0]-1):
            #     image1 = images[i,None]
            #     image2 = images[i+1,None]
            # print("before cuda",im1.shape)
            im0 = im0.to(DEVICE)
            im1 = im1.to(DEVICE)
            #print(im0.shape)
            # print("after cuda",im1.shape)
            flow_low, flow_up = model(im0[:,0,:,:,:], im1[:,0,:,:,:], iters=20, test_mode=True)
        for i in range(flow_up.shape[0]):
            disparity_assessment(
                    np.moveaxis(im0.cpu().numpy()[i,:,:,:],0,2),
                    np.moveaxis(im1.cpu().numpy()[i,:,:,:],0,2),
                    flow_up[i,:,:,:].permute(1,2,0).cpu().numpy(),
                    path_ref[i],
                    path_meas[i],
                    args.out_dir,
            )
        # images = glob.glob(os.path.join(args.path, '*.png')) + \
        #          glob.glob(os.path.join(args.path, '*.jpg'))
        # print("Used memory:",get_gpu_memory()[0])
        #
        #
        # images = load_image_list(images)
        # for i in range(images.shape[0]-1):
        #     image1 = images[i,None]
        #     image2 = images[i+1,None]
        #     print("Used memory:",get_gpu_memory()[0])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--out_dir',default="./alloutputs/out_images", help="path to output dir")
    parser.add_argument('--left',nargs="+", help="Single or list of left images(space seperated) for ex --left /abc/abc.png /123/123.png")
    parser.add_argument('--right',nargs="+", help="Single or list of right image(space seperated) for ex --right /abc/abc.png /123/123.png")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--batch', help="batch_size")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    args = parser.parse_args()
    #print("Starting Inference for "+args.model)
    assert(len(args.left)==len(args.right))
    dataset = KLens(left=args.left, right=args.right)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          shuffle=False,
                                          batch_size=int(args.batch),
                                          num_workers=4,
                                          drop_last=False,
                                          pin_memory=True)
    demo(args,data_loader)
    print("Used memory:",get_gpu_memory()[0])
