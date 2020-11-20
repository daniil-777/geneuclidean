command="-n 10 -W 4:00 -R "rusage[mem=12000,ngpus_excl_p=1]"  -oo logs/atom_num_bio_local_out_resnet_0_random python train_all_folds.py --loc=lab --config=configurations/config_lab/bio_e3nn/bio_local_resnet/bio_local_out_resnet_0.yaml --radious=8 --type_feature=atom_number  --type_filtering=all --h_filterig=-h --type_fold=random"
bsub -J bio_local_out_resnet_0_random $command
for i in {1..6}; do
    bsub -J bio_local_out_resnet_0_random -w "ended(bio_local_out_resnet_0_random)" $command
done