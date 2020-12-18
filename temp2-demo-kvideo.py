import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
from KLens_video2 import KLens
from frame_utils import writeFlow
import flow_viz
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda'


def disparity_assessment(ref_image, neighbor_image, disp,path_ref,path_meas,args):
    # Warping
    root_path = "./"+args.root_path.split("/")[-1]+"/"
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

# def viz(img, flo):
    
    # img = img[0].permute(1,2,0).cpu().numpy()
    # flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    # flo = flow_viz.flow_to_image(flo)
    # img_flo = np.concatenate([img, flo], axis=0)

    # cv2.imwrite('image.jpg', img_flo[:, :, [2,1,0]])
    # cv2.waitKey()


def demo(args,data_loader):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    for idx, sample in enumerate(data_loader):
        with torch.no_grad():
            print(str(idx)+"/"+str(data_loader.__len__())+" | "+str(idx/data_loader.__len__()*100))
            im0, im1, path_ref,path_meas = sample
            # images = load_image_list(imagenames)
            # for i in range(images.shape[0]-1):
            #     image1 = images[i,None]
            #     image2 = images[i+1,None]
            # print("before cuda",im1.shape)
            im0 = im0.to(DEVICE)
            im1 = im1.to(DEVICE)
            # print(im0.shape)
            # print("after cuda",im1.shape)
            flow_low, flow_up = model(im0[:,0,:,:,:], im1[:,0,:,:,:], iters=20, test_mode=True)
        for i in range(flow_up.shape[0]):
            disparity_assessment(
                    np.moveaxis(im0.cpu().numpy()[i,:,:,:],0,2),
                    np.moveaxis(im1.cpu().numpy()[i,:,:,:],0,2),
                    flow_up[i,:,:,:].permute(1,2,0).cpu().numpy(),
                    path_ref[i],
                    path_meas[i],
                    args,
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--batch', help="batch_size")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--root_path', help='path to video frames folder ending without /')
    args = parser.parse_args()
    dataset = KLens(root_path=args.root_path)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          shuffle=False,
                                          batch_size=int(args.batch),
                                          num_workers=4,
                                          drop_last=False,
                                          pin_memory=True)
    demo(args,data_loader)


