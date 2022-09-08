import string
import random

common = {
    "disable_tqdm":True,
    "exp_name": "webnlg_star",#"nyt","nyt_star",webnlg,webnlg_star
    "rel2id": "rel2id.json",
    "device_num": 0,
    # "encoder": "BiLSTM",
    "encoder": "BERT",
    "hyper_parameters": {
        "shaking_type": "cat",
        # cat, cat_plus, cln, cln_plus; Experiments show that cat/cat_plus work better with BiLSTM, while cln/cln_plus work better with BERT. The results in the paper are produced by "cat". So, if you want to reproduce the results, "cat" is enough, no matter for BERT or BiLSTM.
        "inner_enc_type": "lstm",#not change here
        # valid only if cat_plus or cln_plus is set. It is the way how to encode inner tokens between each token pairs. If you only want to reproduce the results, just leave it alone.
        "dist_emb_size": -1,
        # -1: do not use distance embedding; other number: need to be larger than the max_seq_len of the inputs. set -1 if you only want to reproduce the results in the paper.
        "ent_add_dist": False,
        # set true if you want add distance embeddings for each token pairs. (for entity decoder)
        "rel_add_dist": False,  # the same as above (for relation decoder)
        "match_pattern": "whole_text",
        # only_head_text (nyt_star, webnlg_star), whole_text (nyt, webnlg), only_head_index, whole_span
    },
    "training_method":"self-training",#"self-training","self-emsembling","increment-training"
    "use_two_model": False,
    "two_models_hyper_parameters":{
        "student_dropout": 0.4,
        "student_decay": 0,  # 0 equal to not decay
    },
    "use_strategy":True,
    # "strategy_hyper_parameters":{
    #     "Z_RATIO":0.3,
    # },# use this when use_strategy==False
    "strategy_hyper_parameters":{
            "enh_rate":2,
            "relh_rate":2,
        },# use this when use_strategy==True
    "LABEL_OF_TRAIN": 0.2,


}



common["run_name"] = "{}+{}+{}".format("TP1", common["hyper_parameters"]["shaking_type"], common["encoder"]) + ""

run_id = ''.join(random.sample(string.ascii_letters + string.digits, 8))
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
        "batch_size": 24,
        "TOTAL_EPOCHS": 20,
        "epochs": 5,
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

eval_config = {
    "model_state_dict_dir": "./default_log_dir",
    # if use wandb, set "./wandb", or set "./default_log_dir" if you use default logger
    "run_ids": ["DGKhEFlH", ],
    "last_k_model": 1,
    "test_data": "*test*.json",  # "*test*.json"

    # where to save results
    "save_res": True,
    "save_res_dir": "./test_results",

    # score: set true only if test set is annotated with ground truth
    "score": True,

    "hyper_parameters": {
        "batch_size": 24,
        "force_split": False,
        "max_test_seq_len": 512,
        "sliding_len": 50,
    },
}

bert_config = {
    "data_home": "../data4tplinker/data4bert",
    "bert_path": "/home/hcw/bert-base-cased",
    "hyper_parameters": {
        "lr": 4e-5,
    },
}
bilstm_config = {
    "data_home": "../data4tplinker/data4bilstm",
    "token2idx": "token2idx.json",
    "pretrained_word_embedding_path": "/home/hcw/pretrained_emb/glove_300_nyt.emb",
    "hyper_parameters": {
        "lr": 1e-3,
        "enc_hidden_size": 300,
        "dec_hidden_size": 600,
        "emb_dropout": 0.1,
        "rnn_dropout": 0.1,
        "word_embedding_dim": 300,
    },
}

cawr_scheduler = {
    # CosineAnnealingWarmRestarts
    "T_mult": 1,
    "rewarm_epoch_num": 2,
}
step_scheduler = {
    # StepLR
    "decay_rate": 0.999,
    "decay_steps": 100,
}

# ---------------------------dicts above is all you need to set---------------------------------------------------
if common["encoder"] == "BERT":
    hyper_params = {**common["hyper_parameters"], **bert_config["hyper_parameters"]}
    common = {**common, **bert_config}
    common["hyper_parameters"] = hyper_params
elif common["encoder"] == "BiLSTM":
    hyper_params = {**common["hyper_parameters"], **bilstm_config["hyper_parameters"]}
    common = {**common, **bilstm_config}
    common["hyper_parameters"] = hyper_params

hyper_params = {**common["hyper_parameters"], **train_config["hyper_parameters"]}
train_config = {**train_config, **common}
train_config["hyper_parameters"] = hyper_params
if train_config["hyper_parameters"]["scheduler"] == "CAWR":
    train_config["hyper_parameters"] = {**train_config["hyper_parameters"], **cawr_scheduler}
elif train_config["hyper_parameters"]["scheduler"] == "Step":
    train_config["hyper_parameters"] = {**train_config["hyper_parameters"], **step_scheduler}

hyper_params = {**common["hyper_parameters"], **eval_config["hyper_parameters"]}
eval_config = {**eval_config, **common}
eval_config["hyper_parameters"] = hyper_params


save_folder=f'dst_{common["exp_name"]}-mtp_{common["hyper_parameters"]["match_pattern"]}-twomodel_{1 if common["use_two_model"] else 0}-trainingway_{common["training_method"]}-label_{common["LABEL_OF_TRAIN"]}-lr_{train_config["hyper_parameters"]["lr"]}-encoder_{common["encoder"]}-strategy_{2 if common["use_strategy"] else 0}-skt_{common["hyper_parameters"]["shaking_type"]}-batch_{train_config["hyper_parameters"]["batch_size"]}-totalep_{train_config["hyper_parameters"]["TOTAL_EPOCHS"]}-teaep_{train_config["hyper_parameters"]["epochs"]}-stuep_{train_config["hyper_parameters"]["student_epochs"]}'
train_config["log_path"]="./default_log_dir/{}/{}/default.log".format(save_folder,run_id)
train_config["path_to_save_model"]= "./default_log_dir/{}/{}".format(save_folder,run_id)
