command="-n 4 -W 4:00 -R "rusage[ngpus_excl_p=1]" -R volta  -oo logs/e3nn_11  python train_captioning.py  configurations/config_lab/e3nn/e3nn_11.yaml  random 0"
bsub -J job_chain $command
for i in {1..3}; do
    bsub -J job_chain -w "done(job_chain)" $command
done