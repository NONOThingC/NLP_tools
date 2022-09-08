import string
import random
class DynConfig:
    def __init__(self, stat_dict,dyn_dict=None):
        if dyn_dict is None:
            self.config={}
            self.merge_config(self.config,stat_dict)
        else:
            self.merge_config(stat_dict,dyn_dict)
            self.config=stat_dict

    def merge_config(self,c_a,c_b):
        c_a.update(c_b)
        self.config=c_a

 
common = {
    "disable_tqdm":True,
    "exp_name": "webnlg_star",#"nyt","nyt_star",webnlg,webnlg_star
    "rel2id": "rel2id.json",
    "device_num": 0,
    # "encoder": "BiLSTM",
    "encoder": "BERT",
    "hyper_parameters": {
        "shaking_type": "cln_plus",
        # cat, cat_plus, cln, cln_plus; Experiments show that cat/cat_plus work better with BiLSTM, while cln/cln_plus work better with BERT. The results in the paper are produced by "cat". So, if you want to reproduce the results, "cat" is enough, no matter for BERT or BiLSTM.
        "inner_enc_type": "lstm",#not change here
        # valid only if cat_plus or cln_plus is set. It is the way how to encode inner tokens between each token pairs. If you only want to reproduce the results, just leave it alone.
        "dist_emb_size": -1,
        # -1: do not use distance embedding; other number: need to be larger than the max_seq_len of the inputs. set -1 if you only want to reproduce the results in the paper.
        "ent_add_dist": False,
        # set true if you want add distance embeddings for each token pairs. (for entity decoder)
        "rel_add_dist": False,  # the same as above (for relation decoder)
        "match_pattern": "only_head_text",
        # only_head_text (nyt_star, webnlg_star), whole_text (nyt, webnlg), only_head_index, whole_span
    },
    "training_method":"self-training",#"self-training","self-emsembling","increment-training"
    "use_two_model": False,
    "two_models_hyper_parameters":{
        "student_dropout": 0.4,
        "student_decay": 0,  # 0 equal to not decay
    },
    "use_strategy":False,
     "strategy_hyper_parameters":{
         "Z_RATIO":0.3,
     },# use this when use_strategy==False
    #"strategy_hyper_parameters":{
    #        "enh_rate":2,
    #        "relh_rate":2,
    #    },# use this when use_strategy==True
    "LABEL_OF_TRAIN": 0.3,
}


run_id=''.join(random.sample(string.ascii_letters + string.digits, 8))


common["run_name"] = "{}+{}+{}".format("TP1", common["hyper_parameters"]["shaking_type"], common["encoder"]) + ""


train_config = {
    "train_data": "train_data.json",
    "valid_data": "valid_data.json",
    "rel2id": "rel2id.json",
    # "logger": "wandb", # if wandb, comment the following four lines

    # if logger is set as default, uncomment the following four lines
    "logger": "default",
    "run_id": run_id,
    # "log_path": "./default_log_dir/{}/default.log".format(run_id),
    # "path_to_save_model": "./default_log_dir/{}".format(run_id),

    # only save the model state dict if F1 score surpasses <f1_2_save>
    "f1_2_save": 0,
    # whether train_config from scratch
    "fr_scratch": True,
    # write down notes here if you want, it will be logged
    "note": "start from scratch",
    # if not fr scratch, set a model_state_dict
    "model_state_dict_path": "/home/hcw/TPlinker-joint-extraction-master/semitplinker/default_log_dir/U9Hs8dwu/model_state_dict_teacher_best.pt",
    "hyper_parameters": {
        "batch_size": 16,
        "TOTAL_EPOCHS": 4,
        "epochs": 10,
        "student_epochs": 4,
        "seed": 2333,
        "log_interval": 10,
        "max_seq_len": 100,
        "sliding_len": 20,
        "loss_weight_recover_steps": 6000,
        # to speed up the training process, the loss of EH-to-ET sequence is set higher than other sequences at the beginning, but it will recover in <loss_weight_recover_steps> steps.
        "scheduler": "CAWR",  # Step
    },
}
a=DynConfig(common)
a.merge_config(common,train_config)

