# ✅ 최종 안정화 버전 - 원본 해상도 (765x88) 사용

import tensorflow as tf

# ✅ Metal GPU 완전 비활성화 (Mac M1/M2/M4 대응)
tf.config.set_visible_devices([], 'GPU')

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Concatenate, Lambda
from tensorflow.keras import backend as K
import time
from tqdm import tqdm

# ----------------------------
# 설정
# ----------------------------
image_shape = (88, 765, 1)  # 원본 해상도
handcrafted_dim = 9
csv_path = "/Users/chanyoungko/Desktop/HandWriting/handwriting_pairs_train.csv"
epochs = 10
batch_size = 8  # 안정적 CPU 학습

# ----------------------------
# 유클리드 거리 함수
# ----------------------------
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

# ----------------------------
# 모델 구조
# ----------------------------
def build_cnn_branch(input_shape):
    input_img = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='gelu', padding='same')(input_img)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='gelu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = GlobalAveragePooling2D()(x)  # Flatten 대신 사용
    x = Dense(128, activation='gelu')(x)
    return Model(inputs=input_img, outputs=x)

def build_handcrafted_branch(input_dim):
    input_hand = Input(shape=(input_dim,))
    x = Dense(32, activation='gelu')(input_hand)
    return Model(inputs=input_hand, outputs=x)

def build_siamese_model():
    cnn_branch = build_cnn_branch(image_shape)
    hand_branch = build_handcrafted_branch(handcrafted_dim)

    input_img1 = Input(shape=image_shape)
    input_img2 = Input(shape=image_shape)
    input_hand1 = Input(shape=(handcrafted_dim,))
    input_hand2 = Input(shape=(handcrafted_dim,))

    feat_img1 = cnn_branch(input_img1)
    feat_img2 = cnn_branch(input_img2)
    feat_hand1 = hand_branch(input_hand1)
    feat_hand2 = hand_branch(input_hand2)

    merged_feat1 = Concatenate()([feat_img1, feat_hand1])
    merged_feat2 = Concatenate()([feat_img2, feat_hand2])

    embed1 = Dense(64, activation='gelu')(merged_feat1)
    embed2 = Dense(64, activation='gelu')(merged_feat2)

    distance = Lambda(euclidean_distance)([embed1, embed2])
    model = Model(inputs=[input_img1, input_hand1, input_img2, input_hand2], outputs=distance)
    return model

# ----------------------------
# 특징 추출 함수
# ----------------------------
def extract_handcrafted_features(img):
    features = []
    img_uint8 = img.astype(np.uint8)
    _, binary = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    features.append(np.mean(img) / 255.0)
    features.append(np.std(img) / 255.0)

    edges = cv2.Canny(binary, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features.append(len(contours) / 10.0)

    h, w = img.shape
    features.append(w / h)

    hist = cv2.calcHist([img_uint8], [0], None, [32], [0, 256])
    hist = hist / np.sum(hist)
    features.append(np.sum(hist ** 2))

    hog_descriptor = hog(img_uint8, orientations=8, pixels_per_cell=(16, 16),
                         cells_per_block=(1, 1), visualize=False, feature_vector=True)
    features.append(np.mean(hog_descriptor))
    features.append(np.std(hog_descriptor))

    sobelx = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=5)
    angles = np.arctan2(sobely, sobelx)
    avg_angle = np.mean(angles) / np.pi
    features.append(avg_angle)

    pixel_range = np.max(img_uint8) - np.min(img_uint8)
    contrast_feature = pixel_range / (np.mean(img_uint8) + 1)
    features.append(contrast_feature)

    features = np.clip(features, -1, 1)
    return np.array(features, dtype=np.float32)

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img.astype(np.float32)

# ----------------------------
# 학습 함수
# ----------------------------
def load_and_train_model_verbose():
    start_time = time.time()
    df = pd.read_csv(csv_path)

    img1_list, img2_list = [], []
    hand1_list, hand2_list = [], []
    labels = []

    print("\U0001f9ea 이미지와 특징 추출 중...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="\U0001f50d Loading Data"):
        img1 = preprocess_image(row['image_path_1'])
        img2 = preprocess_image(row['image_path_2'])
        if img1 is None or img2 is None:
            continue
        hand1 = extract_handcrafted_features(img1)
        hand2 = extract_handcrafted_features(img2)
        img1_list.append(np.expand_dims(img1 / 255.0, axis=-1))
        img2_list.append(np.expand_dims(img2 / 255.0, axis=-1))
        hand1_list.append(hand1)
        hand2_list.append(hand2)
        labels.append(row['label'])

    print(f"\n✅ 총 {len(labels)}쌍 데이터 로딩 완료")
    data_loading_time = time.time()
    print(f"⏱️ 데이터 로딩 시간: {data_loading_time - start_time:.2f}초")

    img1_arr = np.array(img1_list)
    img2_arr = np.array(img2_list)
    hand1_arr = np.array(hand1_list)
    hand2_arr = np.array(hand2_list)
    label_arr = np.array(labels).astype(np.float32)

    img1_train, img1_val, img2_train, img2_val, hand1_train, hand1_val, hand2_train, hand2_val, y_train, y_val = train_test_split(
        img1_arr, img2_arr, hand1_arr, hand2_arr, label_arr, test_size=0.2, random_state=42)

    model = build_siamese_model()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("\U0001f680 모델 학습 시작...")
    training_start = time.time()

    history = model.fit(
        [img1_train, hand1_train, img2_train, hand2_train],
        y_train,
        validation_data=([img1_val, hand1_val, img2_val, hand2_val], y_val),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )

    training_end = time.time()
    print(f"\n✅ 모델 학습 완료! ⏱️ 학습 시간: {training_end - training_start:.2f}초")
    print(f"⏱️ 전체 소요 시간: {training_end - start_time:.2f}초")

    model.save("siamese_handcrafted_model.h5")
    return history

# ----------------------------
# 메인 실행
# ----------------------------
if __name__ == "__main__":
    load_and_train_model_verbose()
