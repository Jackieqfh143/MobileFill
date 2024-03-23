export PYTHONPATH=$(dirname $(dirname "$PWD"))
python generate.py --sample 1 --pics 100  --ckpt ./checkpoint/990000.pt --size 512