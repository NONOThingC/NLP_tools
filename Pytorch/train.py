from Batch2Device import batch2device

def train_contrastive(config,logger,model,optimizer,scheduler,loss_func,dataloader,evaluator):
    train_dataloader,valid_dataloader=dataloader
    
    for ep in range(config.train_epoch):
        ## train
        model.train()
        t_ep = time.time()
        # epoch parameter start
        start_lr = optimizer.param_groups[0]['lr']
        batch_cum_loss, total_ent_sample_acc, total_head_rel_sample_acc, total_tail_rel_sample_acc = 0., 0., 0., 0.
        # epoch parameter end
        if not config["disable_tqdm"]:
            train_dataloader=tqdm.tqdm(train_dataloader)
        for batch_ind, batch_train_data in enumerate(train_dataloader):
            # batch parameter start
            t_batch = time.time()
            z = (2 * len(rel2id) + 1)
            steps_per_ep = len(train_dataloader)
            total_steps = hyper_parameters["loss_weight_recover_steps"] + 1  # + 1 avoid division by zero error
            current_step = steps_per_ep * ep + batch_ind
            w_ent = max(1 / z + 1 - current_step / total_steps, 1 / z)
            w_rel = min((len(rel2id) / z) * current_step / total_steps,
                        (len(rel2id) / z))
            loss_weights = {"ent": w_ent, "rel": w_rel}
            # batch parameter end
            
            # move to device
            batch_train_data=batch2device(batch_train_data,config.device)
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # model forward start
            inp_dict={
                
            }
            # or
            inp_lst=[batch_input_ids,
                    batch_attention_mask,
                    batch_token_type_ids]

            m_out = model(*model_lst or **inp_dict)
            # model forward end
            
            # grad operation start
            ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs=m_out

            w_ent, w_rel = loss_weights["ent"], loss_weights["rel"]
            loss = w_ent * loss_func(
                ent_shaking_outputs, batch_ent_shaking_tag) + w_rel * loss_func(
                head_rel_shaking_outputs,
                batch_head_rel_shaking_tag) + w_rel * loss_func(
                tail_rel_shaking_outputs, batch_tail_rel_shaking_tag)

            loss.backward()
            optimizer.step()
            
            # grad operation end
            
            # accuracy calculation start
            ent_sample_acc = metrics.get_sample_accuracy(ent_shaking_outputs,
                                                         batch_ent_shaking_tag)
            head_rel_sample_acc = metrics.get_sample_accuracy(
                head_rel_shaking_outputs, batch_head_rel_shaking_tag)
            tail_rel_sample_acc = metrics.get_sample_accuracy(
                tail_rel_shaking_outputs, batch_tail_rel_shaking_tag)

            loss, ent_sample_acc, head_rel_sample_acc, tail_rel_sample_acc = loss.item(), ent_sample_acc.item(), head_rel_sample_acc.item(
            ), tail_rel_sample_acc.item()
            

            batch_cum_loss += loss
            total_ent_sample_acc += ent_sample_acc
            total_head_rel_sample_acc += head_rel_sample_acc
            total_tail_rel_sample_acc += tail_rel_sample_acc

            batch_avg_loss = batch_cum_loss / (batch_ind + 1)
            avg_ent_sample_acc = total_ent_sample_acc / (batch_ind + 1)
            avg_head_rel_sample_acc = total_head_rel_sample_acc / (batch_ind +
                                                                   1)
            avg_tail_rel_sample_acc = total_tail_rel_sample_acc / (batch_ind +
                                                                   1)
            # accuracy calculation end
            batch_print_format = "\rproject: {}, run_name: {}, Epoch: {}/{}, batch: {}/{}, train_loss: {}, " + "t_ent_sample_acc: {}, t_head_rel_sample_acc: {}, t_tail_rel_sample_acc: {}," + "lr: {}, batch_time: {}, total_time: {} -------------"
            # batch logger and print start
            print(batch_print_format.format(
                experiment_name,
                config["run_name"],
                ep + 1,
                config.train_epoch,
                batch_ind + 1,
                len(train_dataloader),
                batch_avg_loss,
                avg_ent_sample_acc,
                avg_head_rel_sample_acc,
                avg_tail_rel_sample_acc,
                optimizer.param_groups[0]['lr'],
                time.time() - t_batch,
                time.time() - t_ep,
            ),
                end="")
            # batch logger and print end
            
            # change lr
            scheduler.step()
        # epoch logger and print start
        logger.log({
            "train_loss": batch_avg_loss,
            "train_ent_seq_acc": avg_ent_sample_acc,
            "train_head_rel_acc": avg_head_rel_sample_acc,
            "train_tail_rel_acc": avg_tail_rel_sample_acc,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "time": time.time() - t_ep,
        })
        # epoch logger and print start
        
    ## valid
    model.eval()
    t_ep = time.time()
    total_ent_sample_acc, total_head_rel_sample_acc, total_tail_rel_sample_acc = 0., 0., 0.
    for batch_ind, batch_valid_data in enumerate(
            tqdm(valid_dataloader, desc="Validating", disable=config["disable_tqdm"])):
        
            # move to device
            batch_valid_data=batch2device(batch_valid_data,config.device)

        with torch.no_grad():
            # model forward start
            inp_dict={
                
            }
            # or
            inp_lst=[batch_input_ids,
                    batch_attention_mask,
                    batch_token_type_ids]

            m_out = model(*model_lst or **inp_dict)
            # model forward end
            ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = m_out
            
        # accuracy calculation start
        ent_sample_acc = metrics.get_sample_accuracy(ent_shaking_outputs,
                                                     batch_ent_shaking_tag)
        head_rel_sample_acc = metrics.get_sample_accuracy(
            head_rel_shaking_outputs, batch_head_rel_shaking_tag)
        tail_rel_sample_acc = metrics.get_sample_accuracy(
            tail_rel_shaking_outputs, batch_tail_rel_shaking_tag)

        ent_sample_acc, head_rel_sample_acc, tail_rel_sample_acc = ent_sample_acc.item(), head_rel_sample_acc.item(
        ), tail_rel_sample_acc.item()

        total_ent_sample_acc += ent_sample_acc
        total_head_rel_sample_acc += head_rel_sample_acc
        total_tail_rel_sample_acc += tail_rel_sample_acc
        # accuracy calculation end
    # epoch logger and print start
    avg_ent_sample_acc = total_ent_sample_acc / len(valid_dataloader)
    avg_head_rel_sample_acc = total_head_rel_sample_acc / len(valid_dataloader)
    avg_tail_rel_sample_acc = total_tail_rel_sample_acc / len(valid_dataloader)

    log_dict = {
        "val_ent_seq_acc": avg_ent_sample_acc,
        "val_head_rel_acc": avg_head_rel_sample_acc,
        "val_tail_rel_acc": avg_tail_rel_sample_acc,
        "valid epoch time": time.time() - t_ep,
    }
    logger.log(log_dict)
    RS_logger.ts_log().add_scalars("Teacher Valid", {
        "e_acc": avg_ent_sample_acc,
        "h_acc": avg_head_rel_sample_acc,
        "t_acc": avg_tail_rel_sample_acc,
    }, RS_logger.get_cur_ep())
    # epoch logger and print end
    return (avg_ent_sample_acc, avg_head_rel_sample_acc, avg_tail_rel_sample_acc)