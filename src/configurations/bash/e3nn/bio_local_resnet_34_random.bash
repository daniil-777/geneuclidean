command="-n 8 -W 4:00 -R "rusage[mem=10000,ngpus_excl_p=1]"  -oo logs/hot_v_bio_local_resnet_34_random python train_all_folds.py --loc=lab --config=configurations/config_lab/bio_e3nn/bio_local_resnet_34.yaml --radious=8 --type_feature=hot_simple  --type_filtering=all --h_filterig=h --type_fold=random"
bsub -J bio_local_resnet_34_random $command
for i in {1..6}; do
    bsub -J bio_local_resnet_34_random -w "ended(bio_local_resnet_34_random)" $command
done