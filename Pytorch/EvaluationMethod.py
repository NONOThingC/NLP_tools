class MetricsCalculator():
    def __init__(self, handshaking_tagger):
        self.handshaking_tagger = handshaking_tagger
        
    def get_sample_accuracy(self, pred, truth):
        '''
        计算所有抽取字段都正确的样本比例
        即该batch的输出与truth全等的样本比例
        '''
        # (batch_size, ..., seq_len, tag_size) -> (batch_size, ..., seq_len)
        pred_id = torch.argmax(pred, dim = -1)
        # (batch_size, ..., seq_len) -> (batch_size, )，把每个sample压成一条seq
        pred_id = pred_id.view(pred_id.size()[0], -1)
        truth = truth.view(truth.size()[0], -1)

        # (batch_size, )，每个元素是pred与truth之间tag相同的数量
        correct_tag_num = torch.sum(torch.eq(truth, pred_id).float(), dim = 1)

        # seq维上所有tag必须正确，所以correct_tag_num必须等于seq的长度才算一个correct的sample
        sample_acc_ = torch.eq(correct_tag_num, torch.ones_like(correct_tag_num) * truth.size()[-1]).float()
        sample_acc = torch.mean(sample_acc_)
        return sample_acc
    
    def get_rel_cpg(self, sample_list, tok2char_span_list, 
                 batch_pred_ent_shaking_outputs,
                 batch_pred_head_rel_shaking_outputs,
                 batch_pred_tail_rel_shaking_outputs, 
                 pattern = "only_head_text"):
        batch_pred_ent_shaking_tag = torch.argmax(batch_pred_ent_shaking_outputs, dim = -1)
        batch_pred_head_rel_shaking_tag = torch.argmax(batch_pred_head_rel_shaking_outputs, dim = -1)
        batch_pred_tail_rel_shaking_tag = torch.argmax(batch_pred_tail_rel_shaking_outputs, dim = -1)

        correct_num, pred_num, gold_num = 0, 0, 0
        for ind in range(len(sample_list)):
            sample = sample_list[ind]
            text = sample["text"]
            tok2char_span = tok2char_span_list[ind]
            pred_ent_shaking_tag = batch_pred_ent_shaking_tag[ind]
            pred_head_rel_shaking_tag = batch_pred_head_rel_shaking_tag[ind]
            pred_tail_rel_shaking_tag = batch_pred_tail_rel_shaking_tag[ind]

            pred_rel_list = self.handshaking_tagger.decode_rel_fr_shaking_tag(text, 
                                                      pred_ent_shaking_tag, 
                                                      pred_head_rel_shaking_tag, 
                                                      pred_tail_rel_shaking_tag, 
                                                      tok2char_span)
            gold_rel_list = sample["relation_list"]

            if pattern == "only_head_index":
                gold_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["predicate"], rel["obj_tok_span"][0]) for rel in gold_rel_list])
                pred_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["predicate"], rel["obj_tok_span"][0]) for rel in pred_rel_list])
            elif pattern == "whole_span":
                gold_rel_set = set(["{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["subj_tok_span"][1], rel["predicate"], rel["obj_tok_span"][0], rel["obj_tok_span"][1]) for rel in gold_rel_list])
                pred_rel_set = set(["{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["subj_tok_span"][1], rel["predicate"], rel["obj_tok_span"][0], rel["obj_tok_span"][1]) for rel in pred_rel_list])
            elif pattern == "whole_text":
                gold_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"], rel["predicate"], rel["object"]) for rel in gold_rel_list])
                pred_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"], rel["predicate"], rel["object"]) for rel in pred_rel_list])
            elif pattern == "only_head_text":
                gold_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"].split(" ")[0], rel["predicate"], rel["object"].split(" ")[0]) for rel in gold_rel_list])
                pred_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"].split(" ")[0], rel["predicate"], rel["object"].split(" ")[0]) for rel in pred_rel_list])


            for rel_str in pred_rel_set:
                if rel_str in gold_rel_set:
                    correct_num += 1

            pred_num += len(pred_rel_set)
            gold_num += len(gold_rel_set)
        
        return correct_num, pred_num, gold_num
        
    def get_prf_scores(self, correct_num, pred_num, gold_num):
        minimini = 1e-10
        precision = correct_num / (pred_num + minimini)
        recall = correct_num / (gold_num + minimini)
        f1 = 2 * precision * recall / (precision + recall + minimini)
        return precision, recall, f1