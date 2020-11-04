command="-n 8 -W 4:00 -R "rusage[mem=10000,ngpus_excl_p=1]"   -oo logs/mass_charges_bio_3  python train_feature_all.py --config=configurations/config_lab/bio_e3nn/bio_e3nn_3.yaml --radious=8 --type_feature=mass_charges  --type_filtering=all --h_filterig=h --type_fold=random --idx_fold=0"
bsub -J mass_charges_bio_3 $command
for i in {1..5}; do
    bsub -J mass_charges_bio_3 -w "ended(mass_charges_bio_3)" $command
done