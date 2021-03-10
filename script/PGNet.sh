export LANG=en_US.UTF-8
FOLDER=$(pwd)
GPU=$1
TEST=$2
MODEL_NAME=PGNet

DISK_CODE=$(pwd)
DISK_DATA=$(pwd)/eComTag
DISK_RESULT=$(pwd)/output/${MODEL_NAME}
DISK_VECTOR=$(pwd)/vector/word2vec.p

python3 train.py \
        --model $MODEL_NAME \
        --cuda \
        --device_id ${GPU} \
        --test ${TEST} \
        --batch_size 16 \
        --label_smoothing \
        --emb_dim 200 \
        --rnn_hidden_dim 256 \
        --pretrain_emb \
        --pointer_gen \
	--is_coverage \
        --save_path $DISK_RESULT \
	--dataset_path $DISK_DATA \
 
