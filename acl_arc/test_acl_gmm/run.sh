for LEARNING_RATE in 1e-5 
do
for n_word in 1 
do
for n_sent in 1
do
for bs in 2 
do
for SEED in 100
do
for save_step in 1000
do 
for warmup_step in 500
do
    python run_citation_classification.py \
        --model_name_or_path allenai/scibert_scivocab_uncased \
        --model_type bert \
        --task_name ours \
        --do_test \
        --data_dir ../../datasets/data_acl_arc/ \
        --max_seq_length 512 --per_gpu_train_batch_size 1 \
        --learning_rate ${LEARNING_RATE} --num_train_epochs 10 \
        --output_dir result_baseline --seed ${SEED} \
        --classification_type multilabel --overwrite_cache \
        --overwrite_output_dir --gradient_accumulation_steps ${bs} \
         --save_steps ${save_step} --k 0 --logging_steps ${save_step} --evaluate_during_training  --n_iter_sent ${n_sent} --n_iter_word ${n_word} --warmup_steps ${warmup_step}
done
done
done 
done 
done
done 
done
