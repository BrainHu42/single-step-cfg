STEP=80000
hf upload BrianHu42/ckpts "./checkpoints/sscfg_fm_imagenet_k8_8gpus/iter_${STEP}.pth" "sscfg_fm_v2_imagenet_k8_train/iter_${STEP}.pth"
# hf upload BrianHu42/ckpts "./checkpoints/sscfg_imagenet_k8_train/iter_${STEP}.pth" "separate_gm_imagenet_k8_train/iter_${STEP}.pth"
# hf upload BrianHu42/ckpts "./checkpoints/gmflow_imagenet_k8_8gpus/iter_${STEP}.pth" "gmflow_imagenet_k8_train/iter_${STEP}.pth"