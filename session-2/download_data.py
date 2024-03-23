import kaggle
kaggle.api.dataset_download_files('gpreda/chinese-mnist',
                                   path='/home/marc/Escritorio/UPC_AI_Deep_Learning/MLOps/aidl-2024-spring-mlops/session-2',
                                   unzip=True,
                                   force=True,
                                   quiet=False)
