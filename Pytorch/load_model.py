#load existing model
if not config["fr_scratch"]:
    model_state_path = config["model_state_dict_path"]
    modelf1.load_state_dict(torch.load(model_state_path))
    if config["is_load_2"]:
        modelf2.load_state_dict(torch.load(config["student_model_state_dict_path"]))
    print("------------model state {} loaded ----------------".format(
        model_state_path.split("/")[-1]))

def load_checkpoint(model, checkpoint, optimizer, loadOptimizer):
    if checkpoint != 'No':
        print("loading checkpoint...")
        model_dict = model.state_dict()
        modelCheckpoint = torch.load(checkpoint)
        pretrained_dict = modelCheckpoint['state_dict']
        # 过滤操作
        new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        # 打印出来，更新了多少的参数
        print('Total : {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
        model.load_state_dict(model_dict)
        print("loaded finished!")
        # 如果不需要更新优化器那么设置为false
        if loadOptimizer == True:
            optimizer.load_state_dict(modelCheckpoint['optimizer'])
            print('loaded! optimizer')
        else:
            print('not loaded optimizer')
    else:
        print('No checkpoint is included')
    return model, optimizer