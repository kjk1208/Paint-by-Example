python -u warping_train.py \
--logdir models/Paint-by-Example \
--pretrained_model pretrained_models/model-13channel.ckpt \
--base configs/warping.yaml \
--scale_lr False
#--pretrained_model pretrained_models/sd-v1-4-modified-9channel.ckpt \

# python -u main.py \
# --logdir models/Paint-by-Example \
# --pretrained_model models/model.ckpt \
# --base configs/v1.yaml \
# --scale_lr False