import logging

logger = logging.getLogger(__name__)

def recreate_value_path(save_path):
    if os.path.exists(save_path):
        os.system(f'rm -r {save_path}')
        logger.info(f"delete old path {save_path}")
    os.mkdir(save_path)
    if os.path.exists(save_path):
        logger.info(f"create {save_path} success")
        return True
    else:
        logger.info(f"create {save_path} fail")
        return False

def get_logger(filename):
    """Return a logger instance that writes in filename

    Args:
        filename: (string) path to log.txt

    Returns:
        logger: (instance of logger)

    """
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(filename, encoding='utf-8')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    handler.setLevel(logging.DEBUG)
    FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    handler.setFormatter(logging.Formatter(FORMAT))
    stream_handler.setFormatter(logging.Formatter(FORMAT))
    logger.addHandler(handler)
    logger.addHandler(stream_handler)
    return logger


def create_log(filename):
    FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT, filename=filename)


if __name__ == '__main__':
    logger = get_logger('../spu_ave_extract/resource/log/log_test.txt')
    logger.info('test')
    logger.warning('t')
    logger.error('t2')
