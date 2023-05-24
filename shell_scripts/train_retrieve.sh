#Copyright 2022 Tsinghua University
#Author: Hong Liu (liuhong21@mails.tsinghua.edu.cn)
python retrieve_kb.py -mode train\
    -cfg batch_size=8\
    gradient_accumulation_steps=4\
    train_retrieve=True\
    epoch_num=30 device=$1\
    exp_name=retrieve1