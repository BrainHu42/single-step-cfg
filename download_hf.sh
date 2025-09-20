export HF_TOKEN=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
mkdir -p ./data/imagenet
cd ./data/imagenet
huggingface-cli download BrianHu42/imagenet-latents \
  --repo-type model \
  --local-dir . \
  --local-dir-use-symlinks False

# install zstd first
tar --use-compress-program="zstd -d --threads=0" -xf train_cache.tar.zst