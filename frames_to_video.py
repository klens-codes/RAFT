import argparse
import cv2
import numpy as np
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--list', nargs="+", help="use regex of frames")
parser.add_argument('--out', help="directory path + filename of output avi file")
args = parser.parse_args()
print(args.list)
img_array = []
for filename in sorted(args.list):
    print(filename)
    img = cv2.imread(filename)
    # img2 = cv2.imread(filename.replace("2_rotation","2").replace("video_out","videos").replace("maskFlownet","RAFT"))
    # img2 = cv2.imread(filename.replace("_flow","").replace("video_out","videos").replace("maskFlownet","RAFT"))
    height, width, layers = (img.shape[0],img.shape[1],img.shape[2])
    size = (width,height)
    # img_array.append(np.concatenate((img,img2),axis=0))
    img_array.append(img)
    # print("5"+5)


out = cv2.VideoWriter(args.out+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
