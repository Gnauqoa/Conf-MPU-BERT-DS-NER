## Code for "Distantly Supervised Named Entity Recognition via Confidence-Based Multi-Class Positive and Unlabeled Learning" published at ACL 2022
This is the BERT-based implementation for Conf-MPU.

## Note:
As we stated in our paper, the confidence scores in training sets are provided by our BiLSTM-based BNPU model (https://github.com/kangISU/Conf-MPU-DS-NER).

## How to run:
> #### Example 1: train Conf-MPU on CoNLL2003_Dict_1.0
> ```
> python bert_ds_ner.py --dataset=CoNLL2003_Dict_1.0 --risk_type Conf-MPU --m 15 --bert_model=bert-base-cased --task_name=ner --max_seq_length=128 --do_train --num_train_epochs 5 --do_eval --warmup_proportion=0.1
> ```
> #### Example 2: train Conf-MPU on BC5CDR_Dict_1.0
> ```
> python bert_ds_ner.py --dataset=BC5CDR_Dict_1.0 --risk_type Conf-MPU --m 28 --task_name=ner --max_seq_length=128 --do_train --num_train_epochs 5 --do_eval --warmup_proportion=0.1
> ```
## Modification
I added the custom evaluate with trained Conf-MPU-BERT-DS-NER model which can eval your test sentences
The predicted output is in pred_test.txt file
Simply run 
```
python test.py --model pytorch_model.bin --config config.json --vocab vocab.txt
```
## Citation
[Distantly Supervised Named Entity Recognition via Confidence-Based Multi-Class Positive and Unlabeled Learning](https://aclanthology.org/2022.acl-long.498) (Zhou et al., ACL 2022)

