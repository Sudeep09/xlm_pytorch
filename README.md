# xlm_pytorch_transformers
The orignal repository of the work lies here "https://github.com/huggingface/pytorch-transformers".

This repo will help you to run xlm model on your custom task using pytorch transformers.

The training data used for this exercise is available in "https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data"

1. Download the train.csv
2. create a directory dataset >> mkdir dataset
3. Run >> python3 create.py which will convert and split the train.csv into train.tsv and dev.tsv inside the "dataset" directory.
4. To train:
# Training
!python3 run_mycustom.py 
    --model_type xlm \
    --model_name_or_path xlm-mlm-en-2048 \
    --task_name mc \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir ./dataset/ \
    --max_seq_length 64 \
    --save_steps 50000\
    --per_gpu_eval_batch_size=1   \
    --per_gpu_train_batch_size=1   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --eval_all_checkpoints \
    --output_dir /tmp/mc_output/ \
  
