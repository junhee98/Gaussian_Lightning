#!/bin/bash

# SCENES=(bicycle bonsai counter garden kitchen room stump train truck)
# SCENES=(bicycle)
SCENES=(bicycle bonsai counter flowers garden kitchen room stump treehill)
VQ_RATIO=0.6
CODEBOOK_SIZE=8192

for SCENE in "${SCENES[@]}"   # Add more scenes as needed
    do
        IMP_PATH=/root/dev/junhee/ai_framework/LightGaussian/output/${SCENE}_ours/distiled
        INPUT_PLY_PATH=/root/dev/junhee/ai_framework/LightGaussian/output/${SCENE}_ours/distiled/point_cloud/iteration_40000/point_cloud.ply

        SAVE_PATH=/root/dev/junhee/ai_framework/LightGaussian/output/${SCENE}_ours/vectree_quantized

        CMD="CUDA_VISIBLE_DEVICES=1 python vectree/vectree.py \
        --important_score_npz_path ${IMP_PATH} \
        --input_path ${INPUT_PLY_PATH} \
        --save_path ${SAVE_PATH} \
        --vq_ratio ${VQ_RATIO} \
        --codebook_size ${CODEBOOK_SIZE} \
        "
        eval $CMD > "logs_ours/${SCENE}_vectree_quantized.log" 2>&1
    done
wait