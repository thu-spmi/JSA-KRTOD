#Copyright 2022 Tsinghua University
#Author: Hong Liu (liuhong21@mails.tsinghua.edu.cn)
# add pseudo_path=your pseudo path to run this script
python main.py -mode train\
    -cfg batch_size=8\
    origin_batch_size=8\
    gradient_accumulation_steps=4\
    train_retrieve=False\
    gt_db=True\
    gpt=True\
    no_user_intent=True\
    retrieve_kb=False\
    with_pseudo_label=True\
    epoch_num=40 device=$1\
    exp_name=gtdb_pseudo2_new