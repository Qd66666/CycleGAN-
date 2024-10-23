set -ex
python train.py --no_AA_BB --aef relu --gpu_ids 0 --name CycleGAN+ --display_port 8888 --loss_type En+SF --dataroot ./{dataset_dir} --gamma_identity 0 --no_dropout --model Cycle_gan+