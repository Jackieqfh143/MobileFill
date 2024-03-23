export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)/src/MAT
python performance.py \
--dataset_name Celeba-hq \
--model_path ./checkpoints/G-step=552000_lr=0.0001_ema_loss=0.4681.pth \
--mask_type thick_256 \
--target_size 256 \
--total_num 2000 \
--sample_num 3 \
--img_dir "/home/codeoops/CV/InPainting/Inpainting_baseline/compare/results/celeba-hq-256/(thick_256_5.0k)/real_imgs" \
--mask_dir "/home/codeoops/CV/InPainting/Inpainting_baseline/compare/results/celeba-hq-256/(thick_256_5.0k)/masks" \
--save_dir ./results \
--aspect_ratio_kept \
--fixed_size \
--center_crop \
--batch_size 10


#export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)/src/MAT
#python performance.py \
#--dataset_name Place \
#--model_path ./checkpoints/G-step=452000_lr=0.0001_ema_loss=0.469.pth \
#--mask_type thick_512 \
#--target_size 512 \
#--total_num 1000 \
#--sample_num 3 \
#--img_dir "/home/codeoops/CV/InPainting/EFill_release/EFill/dataset/validation/Place/thick_512/imgs" \
#--mask_dir "/home/codeoops/CV/InPainting/EFill_release/EFill/dataset/validation/Place/thick_512/masks" \
#--save_dir ./results \
#--aspect_ratio_kept \
#--fixed_size \
#--center_crop \
#--batch_size 1