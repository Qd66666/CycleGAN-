set -ex
python feat.py --dataroot ./{datasets_dir} --name check_1 --model Cycle_gan+ --phase test --no_dropout --gpu_ids 0 --nef 64 --nef 128
