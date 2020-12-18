eval "$(conda shell.bash hook)"
conda activate /data2/opticalflow/rnd/envs/raft

# left_path = './pins/Hella2_4.PNG2s.png'
# right_path = './pins/Hella2_4.PNG4s.png'

cd /data2/opticalflow/rnd/opticalflow/RAFT

python3 demo-all4-raft_sunil.py "$@"


conda activate /data2/opticalflow/rnd/envs/maskflownet

cd ../maskflownet/MaskFlownet-Pytorch/

# python3 demo_anyimages_sunil.py MaskFlownet.yaml --out_dir /data2/opticalflow/rnd/opticalflow/RAFT/alloutputs/output_allRAFTmodels/MaskFlownet -c 5adNov03-0005_1000000.pth --dataset_cfg klens.yaml -b 1 "$@"
python3 demo_anyimages_sunil.py MaskFlownet.yaml -c 5adNov03-0005_1000000.pth --dataset_cfg klens.yaml -b 1 "$@"

#python3 demo-anyimages
