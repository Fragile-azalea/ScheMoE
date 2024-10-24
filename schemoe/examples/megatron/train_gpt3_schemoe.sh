#!/bin/bash

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# CHECKPOINT_PATH=$1 #<Specify path>
# TENSORBOARD_LOGS_PATH=$2 #<Specify path>
VOCAB_FILE=/gpt2-config/vocab.json #<Specify path to file>/gpt2-vocab.json
MERGE_FILE=/gpt2-config/merges.txt #<Specify path to file>/gpt2-merges.txt
DATA_PATH=/wikipedia/wikipedia_text_document #<Specify path and file prefix>_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 2 
    --hidden-size 1024 
    --num-attention-heads 16 
    --seq-length 1024 
    --max-position-embeddings 1024 
)

TRAINING_ARGS=(
    --micro-batch-size 8 
    --global-batch-size 64
    --disable-bias-linear 
    --train-iters 10000 
    --weight-decay 1e-2 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --lr 0.00015
    --lr-decay-style cosine 
    --min-lr 1.0e-5
    --lr-warmup-fraction .01 
    --lr-decay-iters 320000
    --use-legacy-models
    --transformer-impl local
    --schemoe
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1 
	--pipeline-model-parallel-size 1
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000 
    --eval-interval 1000 
    --eval-iters 10
)


torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    2>&1 | tee logs/schemoe_train.log 
