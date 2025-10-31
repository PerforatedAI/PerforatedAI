# BERT Classification Example

This example shows how to train a BERT model for classification using the PerforatedAI library. It works with both IMDB and SNLI datasets. 

Supported models include: 
* bert-base-uncased
* bert-base-cased
* bert-tiny
* bert-mini
* bert-small
* bert-medium
* roberta-base


## Running

### Baseline models

For training baseline models, make sure you are using the off-the-shelf `transformers` library instead of the PerforatedAI-Transformers version:

    pip install transformers

Usage example for IMDB:

    CUDA_VISIBLE_DEVICES=0 python train_bert.py --model_name "prajjwal1/bert-tiny" --dataset imdb --dsn \
       --seed 0 --num_epochs 100 --batch_size 32 --max_len 512 --lr 1e-5 \
       --hidden_dropout_prob 0.1 --attention_probs_dropout_prob 0.1 \
       --model_save_location ./model_output_IMDB --early_stopping --early_stopping_patience 6 \
       --early_stopping_threshold 0.0 --save_steps 500 --evaluation_strategy epoch \
       --save_strategy epoch --save_total_limit 2 --metric_for_best_model eval_accuracy \
       --greater_is_better True --scheduler_type reduce_lr_on_plateau --scheduler_patience 2 \
       --scheduler_factor 0.5 --scheduler_min_lr 1e-7
       
Usage example for SNLI:

    CUDA_VISIBLE_DEVICES=0 python train_bert.py --model_name "prajjwal1/bert-tiny" --dataset snli --dsn \
       --seed 0 --num_epochs 100 --batch_size 256 --max_len 128 --lr 1e-5 \
       --hidden_dropout_prob 0.1 --attention_probs_dropout_prob 0.1 \
       --model_save_location ./model_output_SNLI --early_stopping --early_stopping_patience 6 \
       --early_stopping_threshold 0.0 --save_steps 500 --evaluation_strategy epoch \
       --save_strategy epoch --save_total_limit 2 --metric_for_best_model eval_accuracy \
       --greater_is_better True --scheduler_type reduce_lr_on_plateau --scheduler_patience 2 \
       --scheduler_factor 0.5 --scheduler_min_lr 1e-7

Usage example for AGNEWS:
    CUDA_VISIBLE_DEVICES=0python /content/train_bert.py --model_name "prajjwal1/bert-tiny" --dataset agnews --dsn \
        --seed 0 --num_epochs 10 --batch_size 32 --max_len 512 --lr 1e-5 \
        --hidden_dropout_prob 0.1 --attention_probs_dropout_prob 0.1 \
        --model_save_location ./model_output_AGNEWS_baseline --early_stopping --early_stopping_patience 6 \
        --early_stopping_threshold 0.0 --save_steps 500 --evaluation_strategy epoch \
        --save_strategy epoch --save_total_limit 2 --metric_for_best_model eval_accuracy \
        --greater_is_better True --scheduler_type linear --scheduler_patience 2 \
        --scheduler_factor 0.5 --scheduler_min_lr 1e-7


### PerforatedAI models

To train a PerforatedAI model, first install the PerforatedAI-Transformers version of the `transformers` library:

    git clone https://github.com/PerforatedAI/PerforatedAI-Transformers.git
    cd PerforatedAI-Transformers
    pip install -e .
    pip install perforatedai


Then, make sure your license.yaml file is in this folder, and set your PerforatedAI password:

    export PAIPASSWORD=<your_pai_password>
    export CUDA_VISIBLE_DEVICES=<your_gpu_id>  # important to set this if you have multiple GPUs
    

Usage example for IMDB:

    CUDA_VISIBLE_DEVICES=0 python train_bert_pai.py --model_name "prajjwal1/bert-tiny" --dataset imdb --dsn \
       --pai_save_name my_pai_run --switch_mode DOING_HISTORY \
       --n_epochs_to_switch 10 --p_epochs_to_switch 10 --history_lookback 1\
       --max_dendrites 5 --improvement_threshold 0.0001 \
       --pb_improvement_threshold 0.01 --pb_improvement_threshold_raw 0.001 \
       --unwrapped_modules_confirmed True \
       --seed 0 --num_epochs 100 --batch_size 32 --max_len 512 --lr 1e-5 \
       --hidden_dropout_prob 0.1 --attention_probs_dropout_prob 0.1 \
       --model_save_location ./model_output_PAI_IMDB --early_stopping --early_stopping_patience 6 \
       --early_stopping_threshold 0.0 --save_steps 500 --evaluation_strategy epoch \
       --save_strategy epoch --save_total_limit 2 --metric_for_best_model eval_accuracy --maximizing_score True \
       --greater_is_better True --scheduler_type reduce_lr_on_plateau --scheduler_patience 2 \
       --scheduler_factor 0.5 --scheduler_min_lr 1e-7

Usage example for SNLI:

    CUDA_VISIBLE_DEVICES=0 python train_bert_pai.py --model_name "prajjwal1/bert-tiny" --dataset snli --dsn \
       --pai_save_name my_pai_run --switch_mode DOING_HISTORY \
       --n_epochs_to_switch 10 --p_epochs_to_switch 10 --history_lookback 1\
       --max_dendrites 5 --improvement_threshold 0.0001 \
       --pb_improvement_threshold 0.01 --pb_improvement_threshold_raw 0.001 \
       --unwrapped_modules_confirmed True \
       --seed 0 --num_epochs 100 --batch_size 256 --max_len 128 --lr 1e-5 \
       --hidden_dropout_prob 0.1 --attention_probs_dropout_prob 0.1 \
       --model_save_location ./model_output_PAI_SNLI --early_stopping --early_stopping_patience 6 \
       --early_stopping_threshold 0.0 --save_steps 500 --evaluation_strategy epoch \
       --save_strategy epoch --save_total_limit 2 --metric_for_best_model eval_accuracy --maximizing_score True \
       --greater_is_better True --scheduler_type reduce_lr_on_plateau --scheduler_patience 2 \
       --scheduler_factor 0.5 --scheduler_min_lr 1e-7

Usage example for AGNews:
    CUDA_VISIBLE_DEVICES=0 python train_bert_pai.py --model_name "prajjwal1/bert-tiny" --dataset agnews --pai_save_name my_pai_run\ --switch_mode DOING_HISTORY --n_epochs_to_switch 10 --history_lookback 1 --max_dendrites 5 --improvement_threshold 0.0005\
    --pb_improvement_threshold 0.01 --pb_improvement_threshold_raw 0.001 --unwrapped_modules_confirmed True --seed 0 --num_epochs 10\ --batch_size 64 --max_len 128 --lr 2e-5 --hidden_dropout_prob 0.1 --attention_probs_dropout_prob 0.1 --model_save_location ./ \ model_output_PAI_AGNEWS --early_stopping --early_stopping_patience 3 --early_stopping_threshold 0.0 --evaluation_strategy epoch \ --save_strategy epoch --save_total_limit 5 --metric_for_best_model eval_accuracy --greater_is_better True --scheduler_type linear --scheduler_patience 2 --scheduler_factor 0.5 --scheduler_min_lr 1e-7 --maximizing_score False


Example output graph:

![BERT PAI output](ExamplePAIGraph.png "ExamplePAIGraph")
