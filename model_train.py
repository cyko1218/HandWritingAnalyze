import os
import cv2
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage.feature import hog
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Concatenate, Lambda
from tensorflow.keras import backend as K

# ✅ GPU 사용 비활성화
os.environ["CUDA_VISIBLE_DEVICES"] = ""
tf.config.set_visible_devices([], 'GPU')

# ----------------------------
# 설정
# ----------------------------
image_shape = (88, 765, 1)
handcrafted_dim = 9
csv_path = "/Users/chanyoungko/Desktop/HandWriting/handwriting_pairs_train.csv"
epochs_per_chunk = 10
batch_size = 8
chunk_size = 50000
model_path = "/Users/chanyoungko/Desktop/HandWriting/siamese_handcrafted_model_memory.keras"

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
    x = GlobalAveragePooling2D()(x)
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
# 학습 시간 추적 콜백
# ----------------------------
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.train_start = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        duration = time.time() - self.epoch_start
        elapsed = time.time() - self.train_start
        avg_epoch_time = elapsed / (epoch + 1)
        remaining = avg_epoch_time * (self.params['epochs'] - epoch - 1)
        print(f"⏱️ Epoch {epoch+1} 종료: {duration:.2f}초 소요 | ⏳ 예상 남은 시간: {remaining/60:.2f}분")

# ----------------------------
# 핸드크래프트 특징 추출
# ----------------------------
def extract_handcrafted_features(img):
    features = []
    img_uint8 = img.astype(np.uint8)
    _, binary = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    features.append(np.mean(img) / 255.0)
    features.append(np.std(img) / 255.0)
    hog_features = hog(binary, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, feature_vector=True)
    hog_features = hog_features[:7] if len(hog_features) >= 7 else np.pad(hog_features, (0, 7 - len(hog_features)))
    features.extend(hog_features)
    return np.array(features[:9], dtype=np.float32)

# ----------------------------
# 순차 학습 (청크 번호 지정 지원)
# ----------------------------
def train_model_in_chunks(start_chunk=0):
    full_df = pd.read_csv(csv_path)
    total_rows = len(full_df)
    start_index = start_chunk * chunk_size

    # ✅ 이어 학습 또는 새 모델 생성
    if os.path.exists(model_path):
        print(f"📂 기존 모델 로드: {model_path}")
        model = load_model(model_path, custom_objects={"euclidean_distance": euclidean_distance})
    else:
        print("🆕 새 모델 생성")
        model = build_siamese_model()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    for i in range(start_index, total_rows, chunk_size):
        current_chunk = i // chunk_size
        print(f"\n📦 Chunk {current_chunk + 1}: {i} ~ {min(i + chunk_size, total_rows)}행 학습 시작")
        df = full_df.iloc[i:i+chunk_size]

        img1_list, img2_list, hand1_list, hand2_list, labels = [], [], [], [], []

        for _, row in tqdm(df.iterrows(), total=len(df)):
            img1 = cv2.imread(row['image_path_1'], cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(row['image_path_2'], cv2.IMREAD_GRAYSCALE)
            if img1 is None or img2 is None:
                continue

            img1 = cv2.resize(img1, (765, 88)).astype(np.float32) / 255.0
            img2 = cv2.resize(img2, (765, 88)).astype(np.float32) / 255.0

            hand1 = extract_handcrafted_features(img1 * 255.0)
            hand2 = extract_handcrafted_features(img2 * 255.0)

            img1_list.append(np.expand_dims(img1, axis=-1))
            img2_list.append(np.expand_dims(img2, axis=-1))
            hand1_list.append(hand1)
            hand2_list.append(hand2)
            labels.append(row['label'])

        if len(labels) == 0:
            continue

        img1_arr = np.array(img1_list)
        img2_arr = np.array(img2_list)
        hand1_arr = np.array(hand1_list)
        hand2_arr = np.array(hand2_list)
        label_arr = np.array(labels).astype(np.float32)

        img1_train, img1_val, img2_train, img2_val, hand1_train, hand1_val, hand2_train, hand2_val, y_train, y_val = train_test_split(
            img1_arr, img2_arr, hand1_arr, hand2_arr, label_arr, test_size=0.2, random_state=42)

        history = model.fit(
            [img1_train, hand1_train, img2_train, hand2_train],
            y_train,
            validation_data=([img1_val, hand1_val, img2_val, hand2_val], y_val),
            batch_size=batch_size,
            epochs=epochs_per_chunk,
            verbose=1,
            callbacks=[TimeHistory()]
        )

        model.save(model_path)
        print(f"✅ 모델 저장 완료: {model_path}")

# ----------------------------
# 실행
# ----------------------------
if __name__ == "__main__":
    # 시작할 청크 번호 지정 (예: 0부터 시작하거나 2부터 시작 등)
    train_model_in_chunks(start_chunk=1)
