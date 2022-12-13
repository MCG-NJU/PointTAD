# MultiTHUMOS training script
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=11302 --use_env main.py --dataset multithumos 

# Charades training script
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=11302 --use_env main.py --dataset charades 

