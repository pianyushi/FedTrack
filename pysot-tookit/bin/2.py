import zipfile
import os


def unzip_files(folder_path):
    # 遍历指定文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        # 如果是 ZIP 文件，则解压缩
        if file_name.endswith('.zip'):
            file_path = os.path.join(folder_path, file_name)

            # 创建解压结果目录，与 ZIP 文件同名
            unzip_folder = "E:\LaSOTTest\LaSOTTest"
            # 打开 ZIP 文件并解压缩
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(unzip_folder)


# 执行批量解压函数
unzip_files('E:\LaSOTTest\zip')
