import kaggle
kaggle.api.dataset_download_files('gpreda/chinese-mnist',
                                   path='/home/mramon/Escritorio/AI_Deep_Learning_UPC/MLOPs/aidl-2024-spring-mlops/session-2',
                                   unzip=True,
                                   force=True,
                                   quiet=False)
