#!/bin/bash
#SBATCH --job-name=mattergen_finetune
#SBATCH --partition=GPU # adjust to your GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:2 # request 1 GPU
#SBATCH --time=48:00:00 # adjust max runtime
#SBATCH --output=mattergen_%j.out
#SBATCH --error=mattergen_%j.err
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=EMAIL_PLACEHOLDER

module purge
module load cuda/11.8
ulimit -s unlimited
ulimit -s
# Activate your Python environment

source ~/.bashrc
conda activate mattergen

cd $HOME/SOFT/mattergen_test/

# W&B setup (offline mode)
export WANDB_PROJECT="crystal-generation"
export WANDB_JOB_TYPE="train"
export WANDB_NAME="is_electride_ddp2"
export WANDB_MODE=offline # <-- offline logging
# wandb sync path/to/wandb/offline-run-<ID>
# wandb sync ./wandb

# Export the property to fine-tune
export PROPERTY=is_electride
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mattergen-finetune \
    adapter.pretrained_name=mattergen_base \
    data_module=mp_20 \
    +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY \
    data_module.properties=[$PROPERTY] \
    trainer.strategy=ddp_find_unused_parameters_true \
    trainer.devices=2 \
    trainer.num_nodes=1 \
    trainer.accelerator=gpu \
    trainer.precision=32 \
    trainer.accumulate_grad_batches=4 \
    trainer.max_epochs=500 \
    data_module.num_workers.train=2 \
    data_module.num_workers.val=2
