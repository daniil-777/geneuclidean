command="-n 10 -W 4:00 -R "rusage[ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" -R "select[gpu_mtotal0>=10240]"  -oo logs/e3nn_17_rxx  python train_captioning.py  configurations/config_lab/e3nn/e3nn_17.yaml  random 0"
bsub -J e3nn_17_rtx $command
for i in {1..3}; do
    bsub -J e3nn_17_rtx -w "ended(e3nn_17_rtx)" $command
done