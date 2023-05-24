#Copyright 2022 Tsinghua University
#Author: Hong Liu (liuhong21@mails.tsinghua.edu.cn)
python main.py -mode test\
    -cfg device=$1\
    gpt=False\
    t5_path=$2\
    gt_db=False\
    retrieve_kb=True