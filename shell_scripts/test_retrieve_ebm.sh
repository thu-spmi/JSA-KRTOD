python main.py -mode test\
    -cfg device=$1\
    gpt_path=$2\
    gt_db=False\
    retrieve_kb=True
    rescore_for_generation=True