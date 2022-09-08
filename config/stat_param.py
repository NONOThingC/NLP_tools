
# class Config(ConfigParser):
#     def __init__(self, config_file, file_type="ConfigParser"):
#         if file_type="ConfigParser":
#             raw_config = ConfigParser()
#             raw_config.read(config_file)
#             self.cast_values(raw_config)
#         elif file_type="Json":
#
#     def cast_values(self, raw_config):
#         for section in raw_config.sections():
#             for key, value in raw_config.items(section):
#                 val = None
#                 if type(value) is str and value.startswith("[") and value.endswith("]"):
#                     val = eval(value)
#                     setattr(self, key, val)
#                     continue
#                 for attr in ["getint", "getfloat", "getboolean"]:
#                     try:
#                         val = getattr(raw_config[section], attr)(key)
#                         break
#                     except:
#                         val = value
#                 setattr(self, key, val)

class StatConfig:
    def __init__(self, config_file, file_type="ConfigParser"):
        if file_type=="ConfigParser":
            from configparser import ConfigParser
            raw_config = ConfigParser()
            raw_config.read(config_file)
            self.cast_values(raw_config)
            self.config=raw_config
        elif file_type=="Json":
            import json
            with open(config_file,"r",encoding="utf-8") as f:
                self.config=json.load(f)
        self.file_type=file_type

    def cast_values(self, raw_config):
        for section in raw_config.sections():
            for key, value in raw_config.items(section):
                val = None
                if type(value) is str and value.startswith("[") and value.endswith("]"):
                    val = eval(value)
                    setattr(self, key, val)
                    continue
                for attr in ["getint", "getfloat", "getboolean"]:
                    try:
                        val = getattr(raw_config[section], attr)(key)
                        break
                    except:
                        val = value
                setattr(self, key, val)
    
    def get_dict(self):
        t_d={}
        if self.file_type=="ConfigParser":
            for section in self.config.sections():
                for key, value in self.config.items(section):
                    t_d[key]=value
        elif self.file_type=="Json":
        return t_d
    
if __name__ == "__main__":
    config=StatConfig("config.ini")
    a=config.get_dict()
    print(a)
  
    b = ConfigParser()
    b.read("config.ini")