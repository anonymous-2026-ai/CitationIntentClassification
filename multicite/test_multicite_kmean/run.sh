for LEARNING_RATE in 1e-6 2e-6 3e-6 4e-6 5e-6 1e-5 2e-5 
do
for n_word in 1 2 3 
do
for n_sent in 1 2 3 
do
for bs in 2 4 8 16 32
do
for SEED in 0
do
for save_step in 1000
do 
for warmup_step in 0 500 1000 2000
do
for wd in 0 0.001 0.01 0.02 0.05 0.1
do
    python run_citation_classification.py \
        --model_name_or_path allenai/scibert_scivocab_uncased \
        --model_type bert \
        --task_name ours \
        --do_test \
        --data_dir ../../datasets/data_multicite/ \
        --max_seq_length 512 --per_gpu_train_batch_size 1 \
        --learning_rate ${LEARNING_RATE} --num_train_epochs 10 \
        --output_dir result_baseline --seed ${SEED} \
        --classification_type multilabel --overwrite_cache \
        --overwrite_output_dir --gradient_accumulation_steps ${bs} \
         --save_steps ${save_step} --k 0 --logging_steps ${save_step} --evaluate_during_training  --n_iter_sent ${n_sent} --n_iter_word ${n_word} --warmup_steps ${warmup_step} --weight_decay ${wd}
done 
done
done
done 
done 
done
done 
done
