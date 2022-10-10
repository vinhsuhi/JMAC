for version in V1 V2
do 
for dataname in D_W D_Y EN_DE EN_FR
do 
python main_with_args.py --training_data OpenEA_dataset_v1.1/${dataname}_15K_${version}
done 
done 
