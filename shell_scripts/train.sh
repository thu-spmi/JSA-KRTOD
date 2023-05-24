#Copyright 2022 Tsinghua University
#Author: Hong Liu (liuhong21@mails.tsinghua.edu.cn)
python main.py -mode train\
    -cfg batch_size=8\
    origin_batch_size=8\
    gradient_accumulation_steps=4\
    train_retrieve=False\
    retrieve_kb=False\
    epoch_num=40 device=$1\
    exp_name=gtdb