import os
import argparse
import code
parser = argparse.ArgumentParser()
parser.add_argument('--left',nargs="+", help="restore checkpoint")
parser.add_argument('--right',nargs="+", help="restore checkpoint")
parser.add_argument('--out_dir',nargs="+", help="restore checkpoint")
args = parser.parse_args()
print(args)

left_string = ""
left_folder = []

for count, filename in enumerate(args.left):
    left_string = left_string+filename+" "
    left_folder.append(args.left[0].split('/')[-1])
    if not os.path.exists(left_folder[count]):
        os.makedirs(left_folder[count])


right_string = ""
right_folder = []
for i in args.right:
    right_string = right_string+i+" "
    right_folder.append(args.right[0].split('/')[-1])


# os.system("python3 demo_anyimage_sunil.py --out_dir "+left_folder[count]+" --model checkpoints-20Oct2020-lr10_-6_-finetuneChairSDHom/15000_raft-chairsSDHom.pth --left "+left_string+"--right "+right_string+"--batch 1")
# os.system("python3 demo_anyimage_sunil.py --out_dir "+left_folder[count]+" --model checkpoints-Nov9-ftThings3D/15000_raft-FT-FlyingThings3d.pth --left "+left_string+"--right "+right_string+"--batch 1")
# os.system("python3 demo_anyimage_sunil.py --out_dir "+left_folder[count]+" --model checkpoints-2Nov2020-ftMixture/15000_raft-mixThingswithChairsSDHom.pth --left "+left_string+"--right "+right_string+"--batch 1")
# os.system("python3 demo_anyimage_sunil.py --out_dir "+left_folder[count]+" --model models/raft-sintel.pth --left "+left_string+"--right "+right_string+"--batch 1")
# os.system("python3 demo_anyimage_sunil.py --out_dir "+left_folder[count]+" --model models/raft-kitti.pth --left "+left_string+"--right "+right_string+"--batch 1")
# os.system("python3 demo_anyimage_sunil.py --out_dir "+left_folder[count]+" --model models/raft-things.pth --left "+left_string+"--right "+right_string+"--batch 1")
# code.interact(local=locals())
os.system("python3 demo_anyimage_sunil.py --out_dir "+args.out_dir[0]+" --model checkpoints-20Oct2020-lr10_-6_-finetuneChairSDHom/15000_raft-chairsSDHom.pth --left "+left_string+"--right "+right_string+"--batch 1")
os.system("python3 demo_anyimage_sunil.py --out_dir "+args.out_dir[0]+" --model checkpoints-Nov9-ftThings3D/15000_raft-FT-FlyingThings3d.pth --left "+left_string+"--right "+right_string+"--batch 1")
os.system("python3 demo_anyimage_sunil.py --out_dir "+args.out_dir[0]+" --model checkpoints-2Nov2020-ftMixture/15000_raft-mixThingswithChairsSDHom.pth --left "+left_string+"--right "+right_string+"--batch 1")
os.system("python3 demo_anyimage_sunil.py --out_dir "+args.out_dir[0]+" --model models/raft-sintel.pth --left "+left_string+"--right "+right_string+"--batch 1")
os.system("python3 demo_anyimage_sunil.py --out_dir "+args.out_dir[0]+" --model models/raft-kitti.pth --left "+left_string+"--right "+right_string+"--batch 1")
os.system("python3 demo_anyimage_sunil.py --out_dir "+args.out_dir[0]+" --model models/raft-things.pth --left "+left_string+"--right "+right_string+"--batch 1")    