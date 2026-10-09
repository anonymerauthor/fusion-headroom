#!/usr/bin/env bash
# Parallel version of deep_pool.sh: the 15 members x 2 data sets as a job list, P jobs in parallel on one GPU (resumable).
cd "$(dirname "$0")/.."
P=${P:-3}
mkdir -p runs/deep/cifar10 runs/deep/cifar100
{ for ds in cifar10 cifar100; do sed "s/^/--dataset $ds /" scripts/deep_jobs.txt; done; } |
  xargs -P "$P" -L 1 sh -c 'python scripts/deep_train.py "$@" >> runs/deep/train_$(echo "$@" | cut -d" " -f2).log 2>&1' _
echo ALL_DONE >> runs/deep/pool.status
