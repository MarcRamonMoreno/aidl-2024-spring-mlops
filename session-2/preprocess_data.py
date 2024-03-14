import os
import glob

# Define the dataset directory
dataset_dir = '/home/mramon/Escritorio/AI_Deep_Learning_UPC/MLOPs/aidl-2024-spring-mlops/session-2'

# Find the chinese_mnist.csv file
csv_files = glob.glob(os.path.join(dataset_dir, '*.csv'))

# Optionally, remove other files if you want to keep only the CSV file
for file in csv_files:
    if not file.endswith('chinese_mnist.csv'):
        os.remove(file)