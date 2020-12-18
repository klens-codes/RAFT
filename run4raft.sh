
eval "$(conda shell.bash hook)"
conda activate /data2/opticalflow/rnd/envs/raft

cd /data2/opticalflow/rnd/opticalflow/RAFT

python3 demo-all4-raft.py "$@"


conda activate /data2/opticalflow/rnd/envs/maskflownet

cd ../maskflownet/MaskFlownet-Pytorch/

python3 demo_anyimages.py MaskFlownet.yaml --out_dir /data2/opticalflow/rnd/opticalflow/RAFT/alloutputs/output_allRAFTmodels/MaskFlownet -c 5adNov03-0005_1000000.pth --dataset_cfg klens.yaml -b 1 "$@"


#python3 demo-anyimages
