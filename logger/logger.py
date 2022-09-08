import logging
from functools import wraps

# https://juejin.cn/post/6844904187046658062
class MyLogger:
    def process(func):
        """
        自定义日志处理, 往日志中输出额外参数字段(这里为request id)
        """
        @wraps(func)
        def wrapper(self, msg, *args, **kwargs):
            # 获取调用方所在栈帧(第2帧，数组下标为1)
            frame = inspect.stack()[1]

            # 获取调用方所文件名，这里只取文件名，不带路径
            file_name = os.path.basename(frame[1])

            # 获取代码行数
            file_no = frame[2]

            kwargs["extra"] = {
                # 当前请求id
                "request_id": get_request_id(),
                # 获取调用方模块文件名
                "file_name": file_name,  
                # 获取被调用方法被调用时所处代码行数
                "file_no": file_no
            }
            func(self, msg, *args, **kwargs)
        return wrapper

    def __init__(self, name):
        self.logger = logging.getLogger(name)

    def setLevel(self, log_level):
        self.logger.setLevel(log_level)

    def addHandler(self, handler):
        self.logger.addHandler(handler)

    @process
    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    @process
    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    @process
    def warn(self, msg, *args, **kwargs):
        self.logger.warn(msg, *args, **kwargs)

    @process
    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)





def log_filter(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = 1000 * time.time()
        logger.info(f"=============  Begin: {func.__name__}  =============")
        logger.info(f"Args: {kwargs}")
        try:
            rsp = func(*args, **kwargs)
            logger.info(f"Response: {rsp}")
            end = 1000 * time.time()
            logger.info(f"Time consuming: {end - start}ms")
            logger.info(f"=============   End: {func.__name__}   =============\n")
            return rsp
        except Exception as e:
            logger.error(repr(e))
            raise e
    return wrapper

