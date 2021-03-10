export LANG=en_US.UTF-8
FOLDER=$(pwd)
GPU=$1
TEST=$2
MODEL_NAME=woSSE

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
        --noam \
        --emb_dim 200 \
        --hidden_dim 300 \
        --hop 2 \
        --heads 2 \
        --aln_feature \
        --aln_loss \
        --pretrain_emb \
        --pointer_gen \
        --save_path $DISK_RESULT \
	--dataset_path $DISK_DATA \
 
