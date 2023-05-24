#Copyright 2022 Tsinghua University
#Author: Hong Liu (liuhong21@mails.tsinghua.edu.cn)
#no_user_intent=True\
python main.py -mode train_post\
    -cfg batch_size=8\
    gpt=False\
    gradient_accumulation_steps=4\
    epoch_num=40 device=$1\
    exp_name=t5_post1