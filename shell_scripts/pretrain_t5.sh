#Copyright 2022 Tsinghua University
#Author: Hong Liu (liuhong21@mails.tsinghua.edu.cn)
python main.py -mode pretrain\
    -cfg batch_size=8\
    gpt=False\
    only_target_loss=False\
    save_type=min_loss\
    gradient_accumulation_steps=4\
    epoch_num=30 device=$1\
    exp_name=pretrain_t5