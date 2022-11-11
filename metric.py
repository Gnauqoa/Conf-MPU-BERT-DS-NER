def _eval(newSentences, label_map, args, eval_dataset):
    tag2Idx = {v: k for k, v in label_map.items()}
    trueEntityID, predEntityID = entityIDGeneration(newSentences)

    f1_record = []
    if args.determine_entity:
        labels = []
        preds = []
        for sent in newSentences:
            for token_info in sent:
                labels.append(token_info[1])
                preds.append(token_info[2])
        assert len(labels) == len(preds)
        p, r, f1 = compute_token_f1(labels, preds)
        f1_record.append(f1)
        print("Entity: Precision: {}, Recall: {}, F1: {}".format(p, r, f1))
    else:
        if args.flag == 'ALL' or args.inference:
            flags = [f for f in tag2Idx.keys()][1:-2]
            for flag in flags:
                precision, recall, f1 = compute_precision_recall_f1(trueEntityID, predEntityID, flag, tag2Idx[flag])
                print(flag + ": Precision: {}, Recall: {}, F1: {}".format(precision, recall, f1))
            overall_precision, overall_recall, overall_f1 = compute_overall_precision_recall_f1(trueEntityID, predEntityID, tag2Idx)
            f1_record.append(overall_f1)
            print("OVERALL: Precision: {}, Recall: {}, F1: {}".format(overall_precision, overall_recall, overall_f1))
        else:
            p, r, f1 = compute_precision_recall_f1(trueEntityID, predEntityID, args.flag, 1)
            f1_record.append(f1)
            print(args.flag + ": Precision: {}, Recall: {}, F1: {} on {}".format(p, r, f1, eval_dataset))

    return sum(f1_record)


def entityIDGeneration(sentences):
    sent_id = 0
    type_ = "#"
    flag = -1

    label_start_id = 0
    pred_start_id = 0

    true_entities = []
    pred_entities = []
    for sentence in sentences:
        # print("sentence")
        # print(sentence)
        pre_label = "O"
        sent_true_entities = []
        sent_pred_entities = []
        for i, (word, label, pred) in enumerate(sentence):
            if label == "O":
                if not pre_label == "O":
                    label_end_id = i - 1
                    # print("entity label: ", sent_id, label_start_id, label_end_id, type)
                    sent_true_entities.append("_".join([str(i) for i in [sent_id, label_start_id, label_end_id]] + [type_]))
            else:
                if "B-" in label:
                    label = label.split("-")[-1]
                    if not pre_label == "O":
                        label_end_id = i - 1
                        sent_true_entities.append("_".join([str(i) for i in [sent_id, label_start_id, label_end_id]] + [type_]))
                    label_start_id = i
                    type_ = label
                else:
                    continue
            pre_label = label
        if not pre_label == "O":
            label_end_id = len(sentence) - 1
            # print("entity label: ", sent_id, label_start_id, label_end_id, type)
            sent_true_entities.append("_".join([str(i) for i in [sent_id, label_start_id, label_end_id]] + [type_]))

        pre_pred = 1
        for i, (word, label, pred) in enumerate(sentence):
            if pred == 1:
                if not pre_pred == 1:
                    pred_end_id = i - 1
                    # print("entity pred: ", sent_id, pred_start_id, pred_end_id, flag)
                    sent_pred_entities.append("_".join([str(i) for i in [sent_id, pred_start_id, pred_end_id, flag]]))
            else:
                if not pre_pred == pred:
                    if not pre_pred == 1:
                        pred_end_id = i - 1
                        sent_pred_entities.append("_".join([str(i) for i in [sent_id, pred_start_id, pred_end_id, flag]]))
                    pred_start_id = i
                    flag = pred
                else:
                    continue
            pre_pred = pred

        if not pre_pred == 1:
            pred_end_id = len(sentence) - 1
            # print("entity pred: ", sent_id, pred_start_id, pred_end_id, flag)
            sent_pred_entities.append("_".join([str(i) for i in [sent_id, pred_start_id, pred_end_id, flag]]))

        sent_id += 1
        true_entities.append(sent_true_entities)
        pred_entities.append(sent_pred_entities)
    return true_entities, pred_entities


def compute_token_f1(labels, preds):
    # recall = tp/(tp + fn)
    # precision = tp/(tp + fp)
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    assert len(labels) == len(preds)
    for i in range(len(labels)):
        if (labels[i].startswith("B") or labels[i].startswith("I")) and preds[i] == 1:
            tp += 1
        elif (labels[i].startswith("B") or labels[i].startswith("I")) and preds[i] == 0:
            fn += 1
        elif labels[i].startswith("O") and preds[i] == 0:
            tn += 1
        elif labels[i].startswith("O") and preds[i] == 1:
            fp += 1
    if tp == 0:
        recall = 0
        precision = 0
    else:
        recall = float(tp) / (float(tp) + float(fn))
        precision = float(tp) / (float(tp) + float(fp))
    if recall == 0 or precision == 0:
        f1 = 0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def compute_precision_recall_f1(true_entities, pred_entities, flag, pflag):
    tp = 0
    np_ = 0
    pp = 0
    for i in range(len(true_entities)):
        sent_true = true_entities[i]
        sent_pred = pred_entities[i]
        for e in sent_true:
            if flag in e:
                np_ += 1
                temp = e.replace(flag, str(pflag))
                if temp in sent_pred:
                    tp += 1
        for e in sent_pred:
            if int(e.split("_")[-1]) == pflag:
                pp += 1
    if pp == 0:
        p = 0
    else:
        p = float(tp) / float(pp)
    if np_ == 0:
        r = 0
    else:
        r = float(tp) / float(np_)
    if p == 0 or r == 0:
        f1 = 0
    else:
        f1 = float(2 * p * r) / float((p + r))
    return p, r, f1


def compute_overall_precision_recall_f1(true_entities, pred_entities, tag2Idx):
    tp = 0
    np_ = len(sum(true_entities, []))
    pp = len(sum(pred_entities, []))
    temp = ' '

    assert len(true_entities) == len(pred_entities)
    for i in range(len(true_entities)):
        sent_true = true_entities[i]
        sent_pred = pred_entities[i]
        for e in sent_true:
            for flag in tag2Idx:
                if flag in e:
                    temp = e.replace(flag, str(tag2Idx[flag]))
            if temp in sent_pred:
                tp += 1
    if pp == 0:
        p = 0
    else:
        p = float(tp) / float(pp)
    if np_ == 0:
        r = 0
    else:
        r = float(tp) / float(np_)
    if p == 0 or r == 0:
        f1 = 0
    else:
        f1 = float(2 * p * r) / float((p + r))
    return p, r, f1
