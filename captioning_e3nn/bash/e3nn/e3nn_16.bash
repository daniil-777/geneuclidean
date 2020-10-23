command="-n 4 -W 24:00 -R "rusage[mem=10000,ngpus_excl_p=1]"   -oo logs/e3nn_16  python train_captioning.py  configurations/config_lab/e3nn/e3nn_16.yaml  random 0"
bsub -J e3nn_16 $command
for i in {1..2}; do
    bsub -J e3nn_16 -w "done(e3nn_16)" $command
done