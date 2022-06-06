# !/bin/bash 

for src_dataset in risec chemu japflow mscorpus
do
  for tgt_dataset in risec chemu japflow mscorpus
  do
    CUDA_VISIBLE_DEVICES=1 python3 rel_classification_info.py --src_dataset $src_dataset --tgt_dataset $tgt_dataset --mode batch_eval --domain tgt --dep 0 --alpha 0.5 --batch_size 18
    CUDA_VISIBLE_DEVICES=1 python3 rel_classification_info.py --src_dataset $src_dataset --tgt_dataset $tgt_dataset --mode batch_eval --domain tgt --dep 1 --gnn rgcn --alpha 0.5 --batch_size 18 --gnn_depth 2
    # CUDA_VISIBLE_DEVICES=5 python3 rel_classification_info.py --src_dataset $src_dataset --tgt_dataset $tgt_dataset --mode batch_eval --domain src --dep 1 --gnn rgat --alpha 1.0 --batch_size 4 --gnn_depth 2
  done
done 







# for (( seed = 0 ; seed <= 2; seed++ )) ### Inner for loop ###
# do
#     for dataset in risec chemu mscorpus
#     do
#         CUDA_VISIBLE_DEVICES=$(($seed*2)) python3 rel_classification_info.py --src_dataset $dataset --tgt_dataset $dataset --mode train --domain src --dep 0 --seed $seed --alpha 1.0 --batch_size 6
#         CUDA_VISIBLE_DEVICES=$(($seed*2)) python3 rel_classification_info.py --src_dataset $dataset --tgt_dataset $dataset --mode train --domain src --dep 1 --gnn rgcn --seed $seed --alpha 1.0 --batch_size 4 --gnn_depth 2
#         CUDA_VISIBLE_DEVICES=$(($seed*2)) python3 rel_classification_info.py --src_dataset $dataset --tgt_dataset $dataset --mode train --domain src --dep 1 --gnn rgat --seed $seed --alpha 1.0 --batch_size 4 --gnn_depth 2
#     done &
#   echo "" #### print the new line ###
# done

