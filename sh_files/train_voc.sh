K_train=21
K_test=21
bsize=32
num_epoch=10
KM_INIT=20
KM_NUM=1
KM_ITER=20
SEED=1
LR=1e-4

mkdir -p results/picie/train/${SEED}

python train_picie.py \
--data_root datasets/VOCdevkit/ \
--save_root results/voc/train/${SEED}/sfa/ \
--pretrain \
--voc \
--repeats 1 \
--lr ${LR} \
--seed ${SEED} \
--num_init_batches ${KM_INIT} \
--num_batches ${KM_NUM} \
--kmeans_n_iter ${KM_ITER} \
--K_train ${K_train} --K_test ${K_test} \
--stuff --thing  \
--batch_size_cluster ${bsize} \
--num_epoch ${num_epoch} \
--res 320 --res1 320 --res2 640 \
