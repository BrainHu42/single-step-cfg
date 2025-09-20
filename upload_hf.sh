export HF_TOKEN=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

STEP=120000
hf upload BrianHu42/ckpts "./checkpoints/sscfg_imagenet_k8_train/iter_${STEP}.pth" "sscfg_imagenet_k8_train/iter_${STEP}.pth"