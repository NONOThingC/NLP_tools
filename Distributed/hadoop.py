import os

def put_data_to_hdfs(output_data_path: str, tar_name: str, hdfs_path: str):
    os.popen(f'tar -zvcf {tar_name} {output_data_path}')
    os.popen(f'hdfs dfs -put {tar_name} {hdfs_path}')


def check_is_save_2_hdfs(tar_name: str, hdfs_path: str):
    res_code = os.system(f'hdfs dfs -ls {hdfs_path}{tar_name}')
    if res_code == 256:
        print("save data fail")
    else:
        print("save data success")