---
license: apache-2.0
tags:
- generated_from_trainer
metrics:
- rouge
model-index:
- name: t5_base_s2_bs64_fp16
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# t5_base_s2_bs64_fp16

This model is a fine-tuned version of [t5-base](https://huggingface.co/t5-base) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 1.9863
- Rouge1: 33.979
- Rouge2: 17.441
- Rougel: 29.6444
- Rougelsum: 29.6424
- Bertscore: 0.8367
- Gen Len: 85.2065

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
- train_batch_size: 32
- eval_batch_size: 16
- seed: 42
- distributed_type: multi-GPU
- num_devices: 4
- gradient_accumulation_steps: 2
- total_train_batch_size: 256
- total_eval_batch_size: 64
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3.0

### Training results

| Training Loss | Epoch | Step | Validation Loss | Rouge1  | Rouge2  | Rougel  | Rougelsum | Bertscore | Gen Len |
|:-------------:|:-----:|:----:|:---------------:|:-------:|:-------:|:-------:|:---------:|:---------:|:-------:|
| No log        | 0.97  | 17   | 3.3691          | 22.74   | 8.6711  | 19.7469 | 19.7079   | 0.8502    | 20.0    |
| 3.9659        | 1.97  | 34   | 2.3301          | 24.8339 | 10.7442 | 21.2816 | 21.3221   | 0.7581    | 20.0    |
| 2.9199        | 2.97  | 51   | 1.9863          | 33.6474 | 17.024  | 29.9469 | 29.9313   | 0.8354    | 20.0    |


### Framework versions

- Transformers 4.25.1
- Pytorch 1.10.0+cu111
- Datasets 2.2.1
- Tokenizers 0.13.2
