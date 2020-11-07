command="-n 8 -W 4:00 -R "rusage[mem=10000,ngpus_excl_p=1]"  -oo logs/bio_all_properties_bio_net_22  python train_feature_all.py --config=configurations/config_lab/bio_e3nn/bio_net_22.yaml --radious=8 --type_feature=bio_all_properties  --type_filtering=all --h_filterig=h --type_fold=random --idx_fold=0"
bsub -J bio_net_22 $command
for i in {1..6}; do
    bsub -J bio_net_22 -w "ended(bio_net_22)" $command
done