# !/bin/bash 

# for (( seed = 0 ; seed <= 2; seed++ )) ### Inner for loop ###
# do
#     for dataset in risec chemu mscorpus japflow
#     do
#         CUDA_VISIBLE_DEVICES=$(($seed*2)) python3 rel_classification_info.py --src_dataset $dataset --tgt_dataset $dataset --mode train --domain src --dep 0 --seed $seed --alpha 0.5 --batch_size 6
#         CUDA_VISIBLE_DEVICES=$(($seed*2)) python3 rel_classification_info.py --src_dataset $dataset --tgt_dataset $dataset --mode train --domain src --dep 1 --gnn rgcn --seed $seed --alpha 0.5 --batch_size 4 --gnn_depth 2
#         # CUDA_VISIBLE_DEVICES=$(($seed*2)) python3 rel_classification_info.py --src_dataset $dataset --tgt_dataset $dataset --mode train --domain src --dep 1 --gnn rgat --seed $seed --alpha 1.0 --batch_size 4 --gnn_depth 2
#     done &
#   echo "" #### print the new line ###
# done


# for (( seed = 0 ; seed <= 2; seed++ )) ### Inner for loop ###
# do
#     for dataset in risec chemu mscorpus
#     do
#         CUDA_VISIBLE_DEVICES=$(($seed*2)) python3 rel_classification_info.py --src_dataset $dataset --tgt_dataset $dataset --mode train --domain src --dep 1 --gnn rgcn --seed $seed --alpha 1.0 --batch_size 4 --gnn_depth 1
#         CUDA_VISIBLE_DEVICES=$(($seed*2)) python3 rel_classification_info.py --src_dataset $dataset --tgt_dataset $dataset --mode train --domain src --dep 1 --gnn rgat --seed $seed --alpha 1.0 --batch_size 4 --gnn_depth 1
#     done &
#   echo "" #### print the new line ###
# done

# Done for RISEC
for (( seed = 0 ; seed <= 2; seed++ )) ### Inner for loop ###
do
    for dataset in risec mscorpus chemu
    do
        CUDA_VISIBLE_DEVICES=$(($seed*2)) python3 fewshot_rel_classification_info.py --src_dataset japflow  --tgt_dataset $dataset --domain src --mode train --dep 0 --seed $seed --alpha 0.5 --batch_size 6 --fewshot 0.05
        CUDA_VISIBLE_DEVICES=$(($seed*2)) python3 fewshot_rel_classification_info.py --src_dataset japflow  --tgt_dataset $dataset --mode train --domain src --dep 1 --gnn rgcn --seed $seed --alpha 0.5 --batch_size 4 --gnn_depth 2 --fewshot 0.05
        CUDA_VISIBLE_DEVICES=$(($seed*2)) python3 fewshot_rel_classification_info.py --src_dataset japflow  --tgt_dataset $dataset --domain src --mode train --dep 0 --seed $seed --alpha 0.5 --batch_size 6 --fewshot 0.1
        CUDA_VISIBLE_DEVICES=$(($seed*2)) python3 fewshot_rel_classification_info.py --src_dataset japflow  --tgt_dataset $dataset --mode train --domain src --dep 1 --gnn rgcn --seed $seed --alpha 0.5 --batch_size 4 --gnn_depth 2 --fewshot 0.1
        CUDA_VISIBLE_DEVICES=$(($seed*2)) python3 fewshot_rel_classification_info.py --src_dataset japflow  --tgt_dataset $dataset --domain src --mode train --dep 0 --seed $seed --alpha 0.5 --batch_size 6 --fewshot 0.2
        CUDA_VISIBLE_DEVICES=$(($seed*2)) python3 fewshot_rel_classification_info.py --src_dataset japflow  --tgt_dataset $dataset --mode train --domain src --dep 1 --gnn rgcn --seed $seed --alpha 0.5 --batch_size 4 --gnn_depth 2 --fewshot 0.2
    done &
  echo "Done for the seeds" #### print the new line ###
done

# # Done for CHEMU
# for (( seed = 0 ; seed <= 2; seed++ )) ### Inner for loop ###
# do
#     for dataset in risec mscorpus japflow
#     do
#         CUDA_VISIBLE_DEVICES=$(($seed*2)) python3 fewshot_rel_classification_info.py --src_dataset chemu --tgt_dataset $dataset --domain src --mode train --dep 0 --seed $seed --alpha 0.5 --batch_size 6 --fewshot 0.05
#         CUDA_VISIBLE_DEVICES=$(($seed*2)) python3 fewshot_rel_classification_info.py --src_dataset chemu --tgt_dataset $dataset --mode train --domain src --dep 1 --gnn rgcn --seed $seed --alpha 0.5 --batch_size 4 --gnn_depth 2 --fewshot 0.05
#         CUDA_VISIBLE_DEVICES=$(($seed*2)) python3 fewshot_rel_classification_info.py --src_dataset chemu --tgt_dataset $dataset --domain src --mode train --dep 0 --seed $seed --alpha 0.5 --batch_size 6 --fewshot 0.1
#         CUDA_VISIBLE_DEVICES=$(($seed*2)) python3 fewshot_rel_classification_info.py --src_dataset chemu --tgt_dataset $dataset --mode train --domain src --dep 1 --gnn rgcn --seed $seed --alpha 0.5 --batch_size 4 --gnn_depth 2 --fewshot 0.1
#         CUDA_VISIBLE_DEVICES=$(($seed*2)) python3 fewshot_rel_classification_info.py --src_dataset chemu --tgt_dataset $dataset --domain src --mode train --dep 0 --seed $seed --alpha 0.5 --batch_size 6 --fewshot 0.2
#         CUDA_VISIBLE_DEVICES=$(($seed*2)) python3 fewshot_rel_classification_info.py --src_dataset chemu --tgt_dataset $dataset --mode train --domain src --dep 1 --gnn rgcn --seed $seed --alpha 0.5 --batch_size 4 --gnn_depth 2 --fewshot 0.2
#         # CUDA_VISIBLE_DEVICES=$(($seed*2)) python3 rel_classification_info.py --src_dataset $dataset --tgt_dataset $dataset --mode train --domain src --dep 1 --gnn rgat --seed $seed --alpha 1.0 --batch_size 4 --gnn_depth 2
#     done &
#   echo "Done for the seeds" #### print the new line ###
# done


# CUDA_VISIBLE_DEVICES=5 python3 fewshot_rel_classification_info.py --src_dataset chemu --tgt_dataset risec --domain src --mode train --dep 0 --seed 0 --alpha 0.5 --batch_size 6 --fewshot 0.05