export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)/src/MAT
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file ./configs/acc_opts/acc_1gpu.yaml \
run.py --configs ./configs/place_train_256.yaml
