
import sys
sys.path.append("data")
from ..data.dataset import TitleDataset
from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_metric
from nltk import word_tokenize
import numpy as np
import torch
import wandb
import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    # Training hyperparameters
    parser.add_argument("--include_topic_card", type=bool)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=64)
    #parser.add_argument('--display_step',type=int, default=2000)
    parser.add_argument('--val_batch_size',type=int, default=32)
    parser.add_argument('--test_batch_size',type=int,default=1)
    # Model hyperparameters
    parser.add_argument('--model_name',type=str, default="facebook/bart-large")
    # Optimizer hyperparameters
    parser.add_argument('--init_lr',type=float, default=3e-4)
    parser.add_argument('--warm_up',type=int, default=600)
    parser.add_argument('--weight_decay',type=float, default=1e-2)
    parser.add_argument('--decay_epoch',type=int, default=0)
    parser.add_argument('--adam_beta1',type=float, default=0.9)
    parser.add_argument('--adam_beta2',type=float, default=0.999)
    parser.add_argument('--adam_eps',type=float, default=1e-12)
    parser.add_argument('--dropout_rate',type=float, default=0.1)
    # Tokenizer hyperparameters
    parser.add_argument('--encoder_max_len', type=int, default=1024)
    parser.add_argument('--decoder_max_len', type=int, default=50)
    parser.add_argument('--vocab_size',type=int, default=51201)
    parser.add_argument('--eos_idx',type=int, default=51200)
    parser.add_argument('--tokenizer_name',type=str, default='t5-base')
    # Checkpoint directory hyperparameters
    #parser.add_argument('--pretrained_weight_path',type=str, default='pretrained_weights')
    parser.add_argument('--finetune_weight_path', type=str, default="paper_title_generation/checkpoint2")
    parser.add_argument('--best_finetune_weight_path',type=str, default='paper_title_generation/checkpoint2')
    # Dataset hyperparameters
    parser.add_argument('--test_output_file_name',type=str, default="paper_title_generation/checkpoint2/test_output.txt")
    
    return parser

if __name__=="__main__":

    parser = get_parser()
    args = parser.parse_args()

    finetune_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    finetune_model.gradient_checkpointing_enable()
    finetune_model = finetune_model.to(device)
    columns = ["History", "Predicted Response", "True Response"]
    table = wandb.Table(columns=columns)

    def compute_metrics(eval_pred):
        bleu = load_metric("bleu")
        rouge = load_metric("rouge")
        predictions, labels = eval_pred
        # history_string = tokenizer.batch_decode(history, skip_special_tokens=True)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        pred_string = decoded_preds
        label_string = decoded_labels
        # Rouge expects a newline after each sentence
        decoded_preds = [word_tokenize(pred) for pred in decoded_preds]
        decoded_labels = [[word_tokenize(label)] for label in decoded_labels]
        
        bleu_result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
        rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        
        results = {"BLEU4": bleu_result['bleu']}
        for k, v in rouge_result.items():
            results[k] = v.mid.fmeasure
        # Extract a few results
        
        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        results["gen_len"] = np.mean(prediction_lens)
        return {k: round(v, 4) for k, v in results.items()}


    finetune_args = Seq2SeqTrainingArguments(
        output_dir = args.finetune_weight_path,
        overwrite_output_dir = True,
        do_train=True,
        do_eval=True,
        do_predict=True,
        evaluation_strategy='steps',
        logging_strategy="steps",
        save_strategy= "steps",
        eval_steps=20,
        logging_steps=20,
        # save_steps=1,
        per_device_train_batch_size = args.train_batch_size,
        per_device_eval_batch_size = args.val_batch_size,
        learning_rate=args.init_lr,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_eps,
        num_train_epochs=args.epoch,
        max_grad_norm=0.1,
        #label_smoothing_factor=0.1,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        # max_steps= ,
        lr_scheduler_type='polynomial',
        #warmup_ratio= ,
        warmup_steps= args.warm_up,
        save_total_limit=1,
        fp16=True,
        seed = 516,
        logging_first_step=True,
        load_best_model_at_end=True,
        predict_with_generate=True,
        prediction_loss_only=False,
        generation_max_length=100,
        generation_num_beams=5,
        metric_for_best_model='loss',
        greater_is_better=False,
        report_to = 'wandb',
        # include_inputs_for_metrics=True,
    )
    train_dataset = TitleDataset("train")
    eval_dataset = TitleDataset("valid")
    test_dataset = TitleDataset("test")

    finetune_trainer = Seq2SeqTrainer(
        model = finetune_model,
        args = finetune_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        tokenizer = tokenizer,
        compute_metrics=compute_metrics
        
    
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    wandb.init(project="title_generation", entity="tutoring-convei")
    wandb.run.name = f"epoch{args.epoch}_lr{str(args.init_lr)}"
    finetune_trainer.train()


    # Save final weights
    finetune_trainer.save_model(args.best_finetune_weight_path)

