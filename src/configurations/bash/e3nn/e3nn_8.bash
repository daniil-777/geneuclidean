command="-n 4 -W 24:00 -R "rusage[mem=10000,ngpus_excl_p=1]"   -oo logs/e3nn_8  python train_captioning.py  configurations/config_lab/e3nn/e3nn_8.yaml  random 0"
bsub -J e3nn_8 $command
for i in {1..3}; do
    bsub -J e3nn_8 -w "done(e3nn_8)" $command
done