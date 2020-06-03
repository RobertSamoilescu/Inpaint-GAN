import os
import numpy as np
import pandas as pd
import re

files = os.listdir('/mnt/storage/workspace/roberts/nuscene/dataset/imgs')
files = [file[:-4] for file in files]
num_files = len(files)
test_size = 999
train_size = num_files - test_size

scenes = []
for file in files:
    search = re.search('scene-(\d*)\.*', file, re.IGNORECASE).group(1)
    scenes.append(int(search))

idx = np.argsort(scenes)
files = list(np.array(files)[idx])

train_files = files[:train_size]
test_files = files[train_size:]

train_csv = pd.DataFrame(train_files, columns=["name"])
test_csv = pd.DataFrame(test_files, columns=["name"])

train_csv.to_csv("/mnt/storage/workspace/roberts/nuscene/dataset/train.csv", index=False)
test_csv.to_csv("/mnt/storage/workspace/roberts/nuscene/dataset/test.csv", index=False)

train_csv = pd.read_csv("/mnt/storage/workspace/roberts/nuscene/dataset/train.csv")
test_csv = pd.read_csv("/mnt/storage/workspace/roberts/nuscene/dataset/test.csv")

