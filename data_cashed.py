# -----------------------------------
# 파일을 일반복으로 패스 설정 (label shuffle)
# -----------------------------------
import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Concatenate, Lambda
from tensorflow.keras import backend as K
from skimage.feature import hog

# -------------------------------------------------
# 파일 재로드 후 라벨을 일반복으로 바꾸기 (설정)
# -------------------------------------------------
csv_path = "/Users/chanyoungko/Desktop/HandWriting/handwriting_pairs.csv"
df = pd.read_csv(csv_path)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
shuffled_csv_path = csv_path.replace(".csv", "_shuffled.csv")
df.to_csv(shuffled_csv_path, index=False)
print(f"✅ 셔플된 CSV 저장 완료: {shuffled_csv_path}")