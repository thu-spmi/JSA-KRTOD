#Copyright 2022 Tsinghua University
#Author: Hong Liu (liuhong21@mails.tsinghua.edu.cn)
#add post to posterior
#experiments/gtdb_with_new_structure/best_model\
#experiments/baseline_post_gtdb/best_post_model\
python main.py -mode train_jsa\
    -cfg origin_batch_size=8\
    batch_size=8\
    gradient_accumulation_steps=4\
    gpt_path=gpt2-chinese\
    posterior_path=experiments/baseline_post_gtdb/best_post_model\
    epoch_num=41 device=$1\
    exp_name=jsa_whole_scratch_3