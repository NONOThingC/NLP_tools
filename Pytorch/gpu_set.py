def set_gpu(logger):
    if config.local_rank == -1 or config.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu")
        #config.n_gpu = torch.cuda.device_count()
        config.n_gpu = 1
    else:
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(config.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        config.n_gpu = 1
    config.device = device
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   config.local_rank, device, config.n_gpu, bool(config.local_rank != -1))