command="bsub -n 10 -R "rusage[mem=20000,ngpus_excl_p=8]" -R "select[gpu_model0==GeForceRTX2080Ti]" -oo logs/e3nn_17_20  python train_captioning.py  configurations/config_lab/e3nn/e3nn_17_b_20.yaml  random 0"
bsub -J e3nn_17_b_20 $command
for i in {1..3}; do
    bsub -J e3nn_17_b_20 -w "ended(e3nn_17_b_20)" $command
done