command="-n 8 -W 4:00 -R "rusage[mem=10000,ngpus_excl_p=1]"  -oo --loc=lab logs/hot_v_bio_local_net_33_morgan python train_all_folds.py --config=configurations/config_lab/bio_e3nn/bio_local_net_33.yaml --radious=8 --type_feature=hot_simple  --type_filtering=all --h_filterig=h --type_fold=morgan"
bsub -J bio_local_net_33_morgan $command
for i in {1..6}; do
    bsub -J bio_local_net_33_morgan -w "ended(bio_local_net_33_morgan)" $command
done