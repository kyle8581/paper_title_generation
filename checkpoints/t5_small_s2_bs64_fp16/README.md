---
license: apache-2.0
tags:
- generated_from_trainer
metrics:
- rouge
model-index:
- name: t5_small_s2_bs64_fp16
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# t5_small_s2_bs64_fp16

This model is a fine-tuned version of [t5-small](https://huggingface.co/t5-small) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 4.2188
- Rouge1: 18.5
- Rouge2: 7.0351
- Rougel: 15.1763
- Rougelsum: 15.1888
- Bertscore: 0.8468
- Gen Len: 99.167

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 64
- eval_batch_size: 16
- seed: 42
- distributed_type: multi-GPU
- num_devices: 4
- gradient_accumulation_steps: 2
- total_train_batch_size: 512
- total_eval_batch_size: 64
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3.0

### Training results

| Training Loss | Epoch | Step | Validation Loss | Rouge1  | Rouge2 | Rougel  | Rougelsum | Bertscore | Gen Len |
|:-------------:|:-----:|:----:|:---------------:|:-------:|:------:|:-------:|:---------:|:---------:|:-------:|
| No log        | 1.0   | 9    | 4.7578          | 18.6499 | 6.3372 | 15.8905 | 15.9347   | 0.8424    | 20.0    |
| No log        | 2.0   | 18   | 4.4648          | 19.0826 | 6.7198 | 16.4849 | 16.5319   | 0.8426    | 20.0    |
| 4.7703        | 3.0   | 27   | 4.2188          | 19.9063 | 7.2081 | 17.125  | 17.1255   | 0.8436    | 20.0    |


### Framework versions

- Transformers 4.25.1
- Pytorch 1.10.0+cu111
- Datasets 2.2.1
- Tokenizers 0.13.2
