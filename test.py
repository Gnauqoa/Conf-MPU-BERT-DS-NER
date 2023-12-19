
import argparse
import torch
from pytorch_transformers import (BertConfig,
                                BertForTokenClassification, BertTokenizer)
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                                TensorDataset)
import torch.nn.functional as F
import metric 
import bert_ds_ner as bds
import os
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
label_list = ["O", "PER", "LOC", "ORG", "MISC", "[CLS]", "[SEP]"]

def generate_prediction(newSentences, pred_file):
    with open(pred_file, 'w', encoding='utf-8') as pf:
        for sent in newSentences:
            for token in sent:
                pf.writelines(token[0] + ' ' + token[1] + ' ' + str(token[2] - 1) + '\n')
            pf.writelines('\n')

def check_param(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')
    
def eval(model):
    processor = bds.NerProcessor()
    tokenizer = BertTokenizer(vocab_file=output_vocab_file, do_lower_case=False)
    # tokenizer = BertTokenizer.from_pretrained('', do_lower_case=True)
    eval_examples = processor.get_test_examples(data_dir='')
    eval_features = bds.convert_examples_to_features(eval_examples, label_list, 128, tokenizer, TRAIN=False)
    
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
    all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
    eval_data = bds.TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids, all_lmask_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=8)
    # breakpoint()
    model.eval()

    y_true = []
    y_pred = []
    label_map = {i: label for i, label in enumerate(label_list, 1)}
    epoch_loss = []
    for input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        valid_ids = valid_ids.to(device)
        label_ids = label_ids.to(device)
        l_mask = l_mask.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, label_ids, valid_ids=valid_ids, attention_mask_label=l_mask)
            risk = bds.Risk('Conf-MPU', 15, 0.5, len(label_list) + 1, [0.05465055176037835, 0.040747270664617107, 0.04923362521547384, 0.02255661253014178])
            loss = risk.risk_on_val(logits, label_ids)
            epoch_loss.append(loss.item())
        logits = torch.argmax(logits, dim=2)
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        input_mask = input_mask.to('cpu').numpy()
        
        for i, label in enumerate(label_ids):
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif label_ids[i][j] == len(label_map):
                    y_pred.append(temp_2)
                    break
                else:
                    temp_2.append(logits[i][j])
        print('Risk on valid set: ', np.mean(np.asarray(epoch_loss)))

    sentences = bds.readfile('test.txt')
    newSentences = []
    for sent, preds in zip(sentences, y_pred):
        assert len(sent[0]) == len(sent[1]) == len(preds)
        newSent = []
        for token, label, pred in zip(sent[0], sent[1], preds):
            newSent.append([token, label, pred])
        newSentences.append(newSent)
    print('num_sents', len(newSentences))
    # print('sent', newSentences[0])

    ###
    pred_file = os.path.join('', 'pred_test.txt')
    generate_prediction(newSentences, pred_file)
    # breakpoint()
    metric._eval(newSentences, label_map, args, '')
            
if __name__ == '__main__':
    # output_model_file = "out_base/Conf-MPU/CoNLL2003_Dict_1.0/CoNLL2003_Dict_1.0/pytorch_model.bin"
    # output_config_file = "out_base/Conf-MPU/CoNLL2003_Dict_1.0/CoNLL2003_Dict_1.0/config.json"
    # output_vocab_file = "out_base/Conf-MPU/CoNLL2003_Dict_1.0/CoNLL2003_Dict_1.0/vocab.txt"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--flag', type=str, default="ALL", help='train.flag.txt')
    parser.add_argument('--determine_entity', type=bool, default=False, help='determine entity or not')
    parser.add_argument('--inference', type=bool, default=False, help='do inference or not')
    parser.add_argument('--model', type=str, help='.bin weight file')
    parser.add_argument('--config', type=str, help='config.json')
    parser.add_argument('--vocab', type=str, help='vocab.txt')
    args = parser.parse_args()
    
    output_model_file = args.model
    output_config_file = args.config
    output_vocab_file = args.vocab
    num_labels = len(label_list) + 1
    args.num_class = num_labels
    
    config = BertConfig.from_json_file(output_config_file)
    model = bds.Ner(config)
    model.to(device)
    state_dict = torch.load(output_model_file)
    model.load_state_dict(state_dict)
    eval(model)
