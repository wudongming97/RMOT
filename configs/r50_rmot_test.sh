#python3 submit_kitti.py \
#--meta_arch motr \
#--dataset_file e2e_rmot \
#--epoch 200 \
#--with_box_refine \
#--lr_drop 100 \
#--lr 2e-4 \
#--lr_backbone 2e-5 \
#--batch_size 1 \
#--sample_mode random_interval \
#--sample_interval 10 \
#--sampler_steps 50 90 150 \
#--sampler_lengths 2 3 4 5 \
#--update_query_pos \
#--merger_dropout 0 \
#--dropout 0 \
#--random_drop 0.1 \
#--fp_ratio 0.3 \
#--query_interaction_layer QIM \
#--extra_track_attn \
#--data_txt_path_train ./datasets/data_path/kitti.train \
#--data_txt_path_val ./datasets/data_path/kitti.train \
#--resume exps/kitti_sampler_3/checkpoint0059.pth \
#--output_dir exps/kitti_sampler_3

python3 submit_bdd100k.py \
--meta_arch motr \
--dataset_file e2e_rmot \
--epoch 200 \
--with_box_refine \
--lr_drop 100 \
--lr 2e-4 \
--lr_backbone 2e-5 \
--batch_size 1 \
--sample_mode random_interval \
--sample_interval 10 \
--sampler_steps 50 90 150 \
--sampler_lengths 2 3 4 5 \
--update_query_pos \
--merger_dropout 0 \
--dropout 0 \
--random_drop 0.1 \
--fp_ratio 0.3 \
--query_interaction_layer QIM \
--extra_track_attn \
--data_txt_path_train ./datasets/data_path/kitti.train \
--data_txt_path_val ./datasets/data_path/kitti.train \
--resume exps/kitti_newloss_2/checkpoint0089.pth \
--output_dir exps/bdd100k_the_persons