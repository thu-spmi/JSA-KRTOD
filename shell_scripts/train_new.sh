#Copyright 2022 Tsinghua University
#Author: Hong Liu (liuhong21@mails.tsinghua.edu.cn)
python main.py -mode train\
    -cfg batch_size=4\
    origin_batch_size=4\
    gradient_accumulation_steps=8\
    train_retrieve=False\
    gt_db=True\
    gpt=True\
    retrieve_kb=False\
    epoch_num=40 device=$1\
    exp_name=gtdb_with_retrieval_mix