command="-n 4 -W 24:00 -R "rusage[mem=10000,ngpus_excl_p=1]"   -oo logs/e3nn_6  python train_captioning.py  configurations/config_lab/e3nn/e3nn_6.yaml  random 0"
bsub -J e3nn_6 $command
for i in {1..3}; do
    bsub -J e3nn_6 -w "ended(e3nn_6)" $command
done