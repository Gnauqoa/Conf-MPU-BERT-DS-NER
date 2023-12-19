from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_transformers import (WEIGHTS_NAME, AdamW, BertConfig,
                                BertForTokenClassification, BertTokenizer,
                                WarmupLinearSchedule)
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                                TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from seqeval.metrics import classification_report

from metric import _eval

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# device = torch.device('cpu')


class Ner(BertForTokenClassification):

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                attention_mask_label=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device=device)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)

        # align to BiLSTM loss computation
        softmax = nn.Softmax(dim=2)
        logits = softmax(logits)
        # print('logits shape', logits.shape)
        # print('logits', logits[0])

        return logits


class Risk(object):
    def __init__(self, risk_type, m, eta, num_class, priors):
        self.risk_type = risk_type
        self.m = m
        self.eta = eta
        self.num_class = num_class
        self.priors = priors

    def compute_risk(self, logits, labels, probs=None):
        risk = 0

        if self.risk_type == 'MPN':
            mask = [self.mask_of_label(labels, i) for i in range(self.num_class)]
            logits_set = [logits.masked_select(torch.from_numpy(mask[i]).bool().cuda()).contiguous().view(-1, self.num_class)
                          for i in range(self.num_class)]

            neg_prior = 1 - sum(self.priors)

            # risk1 = P(+)_risk
            risk1 = sum([self.priors[i - 2] * self.MAE(logits_set[i], np.eye(self.num_class)[i])
                         for i in range(2, self.num_class - 2)])  # index of "O" is 1, and remove [CLS] and [SEP]

            # risk2 = N(-)_risk
            risk2 = neg_prior * self.MAE(logits_set[1], np.eye(self.num_class)[1])

            risk = risk1 * self.m + risk2

        elif self.risk_type == 'MPU':
            mask = [self.mask_of_label(labels, i) for i in range(self.num_class)]
            logits_set = [logits.masked_select(torch.from_numpy(mask[i]).bool().cuda()).contiguous().view(-1, self.num_class)
                          for i in range(self.num_class)]

            # risk1 = P(+)_risk
            risk1 = sum([self.priors[i - 2] * self.MAE(logits_set[i], np.eye(self.num_class)[i])
                         for i in range(2, self.num_class - 2)])

            # risk2 = U(-)_risk
            risk2 = (self.MAE(logits_set[1], np.eye(self.num_class)[1]) -
                     sum([self.priors[i - 2] * self.MAE(logits_set[i], np.eye(self.num_class)[1])
                          for i in range(2, self.num_class - 2)]))

            risk = risk1 * self.m + risk2
            if risk2 < 0:
                risk = - risk2

        elif self.risk_type == 'Conf-MPU':
            l_mask = []
            p_mask = []
            for i in range(self.num_class):
                mask1, mask2 = self.mask_of_label_prob(self.eta, labels, probs, i)
                l_mask.append(mask1)
                p_mask.append(mask2)

            mask = [self.mask_of_label(labels, i) for i in range(self.num_class)]

            logits_set = [logits.masked_select(torch.from_numpy(l_mask[i]).bool().cuda()).contiguous().view(-1, self.num_class)
                          for i in range(self.num_class)]

            logits_set2 = [logits.masked_select(torch.from_numpy(mask[i]).bool().cuda()).contiguous().view(-1, self.num_class)
                           for i in range(self.num_class)]

            prob_set = [sum(ele, []) for ele in p_mask]
            prob_set = [torch.tensor(ele).to(device) for ele in prob_set]

            # U'(-)
            risk1 = self.MAE(logits_set[1], np.eye(self.num_class)[1])
            # P'(-)
            risk2 = sum([self.priors[i - 2] * self.conf_MAE(logits_set[i], np.eye(self.num_class)[1], prob_set[i])
                         for i in range(2, self.num_class - 2)])
            # P(-)
            risk3 = sum([self.priors[i - 2] * self.MAE(logits_set2[i], np.eye(self.num_class)[1])
                         for i in range(2, self.num_class - 2)])
            # P(+)
            risk4 = sum([self.priors[i - 2] * self.MAE(logits_set2[i], np.eye(self.num_class)[i])
                         for i in range(2, self.num_class - 2)])

            negative_risk = risk1
            positive_risk = risk2 - risk3 + risk4
            risk = positive_risk * self.m + negative_risk
            if positive_risk < 0:
                risk = negative_risk

        return risk

    @staticmethod
    def conf_MAE(yPred, yTrue, prob):
        if len(yPred) == 0:
            return torch.FloatTensor([0]).cuda()
        y = torch.from_numpy(yTrue).float().cuda()
        # prob = prob.float().cuda()
        temp = torch.FloatTensor.abs(y - yPred)
        loss = torch.mean((temp * 1 / prob).sum(dim=1) / yTrue.shape[0])
        return loss

    @staticmethod
    def MAE(yPred, yTrue):
        if len(yPred) == 0:
            return torch.FloatTensor([0]).cuda()
        y = torch.from_numpy(yTrue).float().cuda()
        temp = torch.FloatTensor.abs(y - yPred)
        loss = torch.mean(temp.sum(dim=1) / yTrue.shape[0])
        return loss

    def mask_of_label(self, labels, class_elem):
        masks = []
        for s in labels:
            s_mask = []
            for w in s:
                if w == class_elem:
                    s_mask.append([1] * self.num_class)  # [1,1,1,1,1] if class_num = 5
                else:
                    s_mask.append([0] * self.num_class)
            masks.append(s_mask)
        return np.array(masks)

    def mask_of_label_prob(self, eta, labels, probs, class_elem):
        l_masks = []
        p_masks = []
        for s_l, s_p in zip(labels, probs):
            s_mask_l = []
            s_mask_p = []
            for w_l, w_p in zip(s_l, s_p):
                if w_l == class_elem and 1 < w_l < self.num_class - 2 and w_p > eta:
                    s_mask_l.append([1] * self.num_class)  # [1,1,1,1,1] if class_num = 5
                    s_mask_p.append([w_p])
                elif w_l == class_elem and w_l == 1 and w_p <= eta:
                    s_mask_l.append([1] * self.num_class)
                    # s_mask_p.append([w_p])
                else:
                    s_mask_l.append([0] * self.num_class)
                    # s_mask_p.append([w_p])

            l_masks.append(s_mask_l)
            p_masks.append(s_mask_p)
        return np.array(l_masks), p_masks

    def risk_on_val(self, logits, labels):
        mask = [self.mask_of_label(labels, i) for i in range(self.num_class)]

        logits_set = [logits.masked_select(torch.from_numpy(mask[i]).bool().cuda()).contiguous().view(-1, self.num_class)
                      for i in range(self.num_class)]

        risk = sum([self.MAE(logits_set[i], np.eye(self.num_class)[i]) for i in range(1, self.num_class - 2)])

        return risk


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, prob=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.prob = prob


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None, prob_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.prob_id = prob_id


def readfile(filename, has_prob=False):
    """
    read file
    """
    f = open(filename)
    data = []
    if not has_prob:
        sentence = []
        label = []
        for line in f:
            if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                if len(sentence) > 0:
                    data.append((sentence, label))
                    sentence = []
                    label = []
                continue
            splits = line.strip().split(' ')
            sentence.append(splits[0])
            label.append(splits[-1])

        if len(sentence) > 0:
            data.append((sentence, label))
            sentence = []
            label = []
    else:
        sentence = []
        label = []
        prob = []
        for line in f:
            if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                if len(sentence) > 0:
                    data.append((sentence, label, prob))
                    sentence = []
                    label = []
                    prob = []
                continue
            splits = line.strip().split(' ')
            sentence.append(splits[0])
            label.append(splits[-2])
            prob.append(splits[-1])

        if len(sentence) > 0:
            data.append((sentence, label, prob))
            sentence = []
            label = []
            prob = []
    return data


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir, has_prob=False):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, has_prob=False):
        """Reads a tab separated value file."""
        return readfile(input_file, has_prob)


class NerProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir, has_prob=False):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.txt" if not has_prob else "train_prob.txt"), has_prob), "train", has_prob)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        # return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]"]
        return ["O", "PER", "LOC", "ORG", "MISC", "[CLS]", "[SEP]"]

    @staticmethod
    def _create_examples(lines, set_type, has_prob=False):
        examples = []
        if not has_prob:
            for i, (sentence, label) in enumerate(lines):
                guid = "%s-%s" % (set_type, i)
                text_a = ' '.join(sentence)
                text_b = None
                label = label
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        else:
            for i, (sentence, label, prob) in enumerate(lines):
                guid = "%s-%s" % (set_type, i)
                text_a = ' '.join(sentence)
                text_b = None
                label = label
                prob = prob
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, prob=prob))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, TRAIN=None):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list, 1)}

    features = []
    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        tokens = []
        labels = []
        valid = []
        label_mask = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    valid.append(0)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0, 1)
        label_mask.insert(0, 1)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                if TRAIN:
                    label_ids.append(int(labels[i]) + 1)
                else:
                    label_ids.append(label_map[labels[i].split('-')[-1]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(label_map["[SEP]"])

        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [1] * len(label_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_ids,
                          valid_ids=valid,
                          label_mask=label_mask))
    return features


def convert_examples_to_features_(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list, 1)}

    features = []
    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        problist = example.prob
        tokens = []
        labels = []
        valid = []
        label_mask = []
        probs = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            prob_1 = problist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(1)
                    probs.append(prob_1)
                else:
                    valid.append(0)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]
            probs = probs[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        prob_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0, 1)
        label_mask.insert(0, 1)
        label_ids.append(label_map["[CLS]"])
        prob_ids.append(-1)
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(int(labels[i]) + 1)
                prob_ids.append(float(probs[i]))
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(label_map["[SEP]"])
        prob_ids.append(-1)

        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [1] * len(label_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)
            prob_ids.append(-1)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)
            prob_ids.append(-1)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length
        assert len(prob_ids) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id=label_ids,
                            valid_ids=valid,
                            label_mask=label_mask,
                            prob_id=prob_ids))
    return features


def train(model, train_features_reqs, optimizer, scheduler, processor, steps, args):
    train_examples, label_list, args.max_seq_length, tokenizer = train_features_reqs
    num_train_optimization_steps = steps
    if args.risk_type == 'Conf-MPU':
        train_features = convert_examples_to_features_(train_examples, label_list, args.max_seq_length, tokenizer)
    else:
        train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer, TRAIN=True)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    all_valid_ids = torch.tensor([f.valid_ids for f in train_features], dtype=torch.long)
    all_lmask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)

    if args.risk_type == 'Conf-MPU':
        all_prob_ids = torch.tensor([f.prob_id for f in train_features], dtype=torch.float)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids, all_lmask_ids, all_prob_ids)
    else:
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids, all_lmask_ids)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    model.train()

    global_step = 0
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = []
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            if args.risk_type == 'Conf-MPU':
                input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask, prob_ids = batch
                logits = model(input_ids, segment_ids, input_mask, label_ids, valid_ids, l_mask)

                risk = Risk(args.risk_type, args.m, args.eta, args.num_class, args.priors)
                loss = risk.compute_risk(logits, label_ids, probs=prob_ids)
            else:
                input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask = batch
                logits = model(input_ids, segment_ids, input_mask, label_ids, valid_ids, l_mask)

                risk = Risk(args.risk_type, args.m, args.eta, args.num_class, args.priors)
                loss = risk.compute_risk(logits, label_ids)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # tr_loss += loss.item()
            tr_loss.append(loss.item())
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

            if global_step % 20 == 0:
                print('Risk at {} step: {}'.format(global_step, loss.item()))
                eval_features_reqs = (label_list, args.max_seq_length, tokenizer)
                evaluate(model, "valid", processor, eval_features_reqs, args)

        print('Risk on train set', np.mean(np.asarray(tr_loss)))

    # Save a trained model and the associated configuration
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    label_map = {i: label for i, label in enumerate(label_list, 1)}
    model_config = {"bert_model": args.bert_model, "do_lower": args.do_lower_case,
                    "max_seq_length": args.max_seq_length, "num_labels": len(label_list) + 1,
                    "label_map": label_map}
    json.dump(model_config, open(os.path.join(args.output_dir, "model_config.json"), "w"))


def evaluate(model, eval_dataset, processor, eval_features_reqs, args):
    if eval_dataset == "dev" or eval_dataset == "valid":
        eval_examples = processor.get_dev_examples(args.data_dir)
    elif eval_dataset == "test":
        eval_examples = processor.get_test_examples(args.data_dir)
    else:
        raise ValueError("eval on dev or test set only")

    label_list, args.max_seq_length, tokenizer = eval_features_reqs
    eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, TRAIN=False)
    logger.info("***** Running evaluation on {} set *****".format(eval_dataset))
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
    all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids, all_lmask_ids)

    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    model.eval()

    y_true = []
    y_pred = []
    label_map = {i: label for i, label in enumerate(label_list, 1)}
    epoch_loss = []
    for input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        valid_ids = valid_ids.to(device)
        label_ids = label_ids.to(device)
        l_mask = l_mask.to(device)

        with torch.no_grad():
            # logits = model(input_ids, segment_ids, input_mask, valid_ids=valid_ids, attention_mask_label=l_mask)
            logits = model(input_ids, segment_ids, input_mask, label_ids, valid_ids=valid_ids, attention_mask_label=l_mask)
            risk = Risk(args.risk_type, args.m, args.eta, args.num_class, args.priors)
            loss = risk.risk_on_val(logits, label_ids)
            epoch_loss.append(loss.item())

        # logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        logits = torch.argmax(logits, dim=2)
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        input_mask = input_mask.to('cpu').numpy()

        for i, label in enumerate(label_ids):
            # temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif label_ids[i][j] == len(label_map):
                    # y_true.append(temp_1)
                    y_pred.append(temp_2)
                    break
                else:
                    # temp_1.append(label_map[label_ids[i][j]])
                    # temp_2.append(label_map[logits[i][j]])
                    temp_2.append(logits[i][j])

    if eval_dataset == 'dev' or eval_dataset == 'valid':
        print('Risk on valid set: ', np.mean(np.asarray(epoch_loss)))

    sentences = readfile(os.path.join(args.data_dir, eval_dataset + '.txt'))
    newSentences = []
    for sent, preds in zip(sentences, y_pred):
        assert len(sent[0]) == len(sent[1]) == len(preds)
        newSent = []
        for token, label, pred in zip(sent[0], sent[1], preds):
            newSent.append([token, label, pred])
        newSentences.append(newSent)
    print('num_sents', len(newSentences))
    print('sent', newSentences[2])

    if eval_dataset == 'test':
        pred_file = os.path.join(args.data_dir, args.risk_type + '_pred_test.txt')
        generate_prediction(newSentences, pred_file)

    _eval(newSentences, label_map, args, eval_dataset)


def generate_prediction(newSentences, pred_file):
    with open(pred_file, 'w', encoding='utf-8') as pf:
        for sent in newSentences:
            for token in sent:
                pf.writelines(token[0] + ' ' + token[1] + ' ' + str(token[2] - 1) + '\n')
            pf.writelines('\n')


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--dataset",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument('--risk_type', type=str, required=True, help='learning type (MPN, MPU, Conf-MPU)')
    parser.add_argument('--num_class', type=int, help='class number')
    parser.add_argument('--m', type=float, required=True, help='class balance rate')
    parser.add_argument('--eta', type=float, default=0.5, help='threshold for selecting samples')
    parser.add_argument('--priors', help='priors of positive classes')

    parser.add_argument('--flag', type=str, default="ALL", help='train.flag.txt')
    parser.add_argument('--determine_entity', type=bool, default=False, help='determine entity or not')
    parser.add_argument('--inference', type=bool, default=False, help='do inference or not')

    # Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval or not.")
    parser.add_argument("--eval_on",
                        default="test",
                        help="Whether to run eval on the dev set or test set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    args = parser.parse_args()

    processors = {"ner": NerProcessor}

    n_gpu = 0
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    args.output_dir = os.path.join('out_base', args.risk_type, args.dataset)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)

    processor = processors[task_name]()
    # label_list = processor.get_labels()
    if 'CoNLL2003' in args.dataset:
        label_list = ["O", "PER", "LOC", "ORG", "MISC", "[CLS]", "[SEP]"]
    elif 'BC5CDR' in args.dataset:
        label_list = ["O", "Chemical", "Disease", "[CLS]", "[SEP]"]
    else:
        raise Exception('Please check the dataset name!')

    num_labels = len(label_list) + 1
    args.num_class = num_labels

    if 'CoNLL2003' in args.dataset:
        if args.dataset == 'CoNLL2003_Fully':
            args.priors = [0.05465055176037835, 0.040747270664617107, 0.04923362521547384, 0.02255661253014178]
        else:
            args.priors = [0.0314966102568, 0.0376880632424, 0.0354240324761, 0.015502139428]
    elif 'BC5CDR' in args.dataset:
        if args.dataset == 'BC5CDR_Fully':
            args.priors = [0.060108318524160105, 0.060082931370060086]  # true
        else:
            args.priors = [0.0503131404897, 0.0503834263676]    # estimated
    else:
        raise Exception('Please check the dataset name!')

    if 'BC5CDR' in args.dataset:
        args.bert_model = 'biobert'
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    args.data_dir = os.path.join('data', args.dataset)
    train_examples = None
    num_train_optimization_steps = 0
    if args.do_train:
        if args.risk_type == 'Conf-MPU':
            train_examples = processor.get_train_examples(args.data_dir, has_prob=True)
        else:
            train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    # Prepare model
    config = BertConfig.from_pretrained(args.bert_model, num_labels=num_labels, finetuning_task=args.task_name)
    model = Ner.from_pretrained(args.bert_model, from_tf=False, config=config)

    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optimization_steps)

    if args.do_train:
        train_features_reqs = (train_examples, label_list, args.max_seq_length, tokenizer)
        train(model, train_features_reqs, optimizer, scheduler, processor, num_train_optimization_steps, args)
    else:
        # Load a trained model and vocabulary that you have fine-tuned
        model = Ner.from_pretrained(args.output_dir)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)

    model.to(device)

    if args.do_eval:
        eval_features_reqs = (label_list, args.max_seq_length, tokenizer)
        eval_dataset = "test"
        evaluate(model, eval_dataset, processor, eval_features_reqs, args)


if __name__ == "__main__":
    main()
