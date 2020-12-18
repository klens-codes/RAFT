import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from KLens_images import KLens
from frame_utils import writeFlow
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
import subprocess as sp
def get_gpu_memory():
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values



def disparity_assessment(ref_image, neighbor_image, disp,path_ref,path_meas):
    # Warping
    root_path = "./out_images/"
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
    print(flo.cpu().numpy().shape)
    flo = flo[0].permute(1,2,0).cpu().numpy()
    print(flo.shape)
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
    print("Used memory:",get_gpu_memory()[0])
    print("starting inference")

    for idx, sample in enumerate(data_loader):
        with torch.no_grad():
            im0, im1, path_ref,path_meas = sample
            print(path_meas)
            # images = load_image_list(imagenames)
            # for i in range(images.shape[0]-1):
            #     image1 = images[i,None]
            #     image2 = images[i+1,None]
            # print("before cuda",im1.shape)
            im0 = im0.to(DEVICE)
            im1 = im1.to(DEVICE)
            print(im0.shape)
            # print("after cuda",im1.shape)
            flow_low, flow_up = model(im0[:,0,:,:,:], im1[:,0,:,:,:], iters=20, test_mode=True)
        for i in range(flow_up.shape[0]):
            print(
                disparity_assessment(
                    np.moveaxis(im0.cpu().numpy()[i,:,:,:],0,2),
                    np.moveaxis(im1.cpu().numpy()[i,:,:,:],0,2),
                    flow_up[i,:,:,:].permute(1,2,0).cpu().numpy(),
                    path_ref[i],
                    path_meas[i]
                )
            )    # print()

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
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--batch', help="batch_size")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    args = parser.parse_args()
    dataset = KLens(root_path="/data2/opticalflow/KLENS/images")
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          shuffle=False,
                                          batch_size=int(args.batch),
                                          num_workers=4,
                                          drop_last=False,
                                          pin_memory=True)
    demo(args,data_loader)
    print("Used memory:",get_gpu_memory()[0])
