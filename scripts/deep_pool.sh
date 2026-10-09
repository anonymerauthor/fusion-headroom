#!/usr/bin/env bash
# Model pool for the deep-committee experiments (15 members per data set). Resumable: finished members are skipped.
cd "$(dirname "$0")/.."
for ds in cifar10 cifar100; do
  run() { python scripts/deep_train.py --dataset $ds "$@" >> runs/deep/train_$ds.log 2>&1; }
  mkdir -p runs/deep/$ds
  for s in 0 1 2 3 4; do run --arch resnet18 --seed $s --name resnet18_s$s; done   # seed ensemble
  run --arch wrn16_4 --seed 0 --name wrn16_4_s0
  run --arch wrn16_4 --seed 1 --name wrn16_4_s1
  run --arch vgg11 --seed 0 --name vgg11_s0
  run --arch mobilenetv2 --seed 0 --name mobilenetv2_s0
  run --arch densenet --seed 0 --name densenet_s0
  run --arch simplecnn --seed 0 --name simplecnn_s0
  run --arch resnet18 --seed 10 --name resnet18_short --epochs 8               # under-trained
  run --arch resnet18 --seed 11 --name resnet18_frac25 --frac 0.25             # 10k training rows
  run --arch resnet18 --seed 12 --name resnet18_ls --label-smoothing 0.1       # label smoothing
  run --arch resnet18 --seed 13 --name resnet18_mixup --mixup 0.2              # mixup
done
echo ALL_DONE >> runs/deep/train_cifar100.log
