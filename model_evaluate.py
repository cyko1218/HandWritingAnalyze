import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from skimage.feature import hog
from tensorflow.keras import backend as K

# -------------------------------------------------
# 📌 설정
# -------------------------------------------------
image_shape = (88, 765)
handcrafted_dim = 9
csv_path = "/Users/chanyoungko/Desktop/HandWriting/handwriting_pairs.csv"
model_path = "/Users/chanyoungko/Desktop/HandWriting/siamese_handcrafted_model_memory.keras"
chunk_size = 10000

# -------------------------------------------------
# 📌 거리 → 유사도 변환 함수
# -------------------------------------------------
def distance_to_similarity(distance, factor=5):
    return np.exp(-distance * factor)

# -------------------------------------------------
# 📌 Threshold 튜닝 함수
# -------------------------------------------------
def tune_threshold(preds, labels, step=0.01):
    best_threshold = 0.5
    best_f1 = 0.0

    thresholds = np.arange(0.0, 1.01, step)
    accs, f1s = [], []

    for t in thresholds:
        binary_preds = [1 if p > t else 0 for p in preds]
        acc = accuracy_score(labels, binary_preds)
        f1 = f1_score(labels, binary_preds)
        accs.append(acc)
        f1s.append(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold, accs, f1s, thresholds

# -------------------------------------------------
# 📌 Handcrafted feature 추출 함수
# -------------------------------------------------
def extract_handcrafted_features(img):
    features = []
    img_uint8 = img.astype(np.uint8)
    _, binary = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    features.append(np.mean(img) / 255.0)
    features.append(np.std(img) / 255.0)
    hog_features = hog(binary, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                       visualize=False, feature_vector=True)
    hog_features = hog_features[:7] if len(hog_features) >= 7 else np.pad(hog_features, (0, 7 - len(hog_features)))
    features.extend(hog_features)
    return np.array(features[:9], dtype=np.float32)

# -------------------------------------------------
# 📌 평가 + Threshold 튜닝 통합 함수
# -------------------------------------------------
def evaluate_and_tune(csv_path, model_path, chunk_size=10000, start_chunk=0):
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    start_index = start_chunk * chunk_size

    model = load_model(model_path, custom_objects={"euclidean_distance": lambda x: x})
    all_preds, all_labels = [], []
    thresholds_per_chunk = []

    for i in range(start_index, total_rows, chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        preds, labels = [], []
        failed = 0
        chunk_id = i // chunk_size

        print(f"\n🔹 Chunk {chunk_id} 평가 시작: {i} ~ {min(i + chunk_size, total_rows)}")

        for _, row in tqdm(chunk.iterrows(), total=len(chunk), desc=f"🧪 Chunk {chunk_id}"):
            path1, path2, label = row['image_path_1'], row['image_path_2'], int(row['label'])

            img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
            if img1 is None or img2 is None:
                failed += 1
                continue

            img1 = cv2.resize(img1, (765, 88)).astype(np.float32)
            img2 = cv2.resize(img2, (765, 88)).astype(np.float32)
            img1 /= 255.0
            img2 /= 255.0

            hand1 = extract_handcrafted_features(img1 * 255.0)
            hand2 = extract_handcrafted_features(img2 * 255.0)

            input1 = np.expand_dims(img1[..., np.newaxis], axis=0)
            input2 = np.expand_dims(img2[..., np.newaxis], axis=0)
            hand1 = np.expand_dims(hand1, axis=0)
            hand2 = np.expand_dims(hand2, axis=0)

            distance = model.predict([input1, hand1, input2, hand2], verbose=0)[0][0]
            similarity = distance_to_similarity(distance)

            preds.append(similarity)
            labels.append(label)

        # Threshold 튜닝
        best_t, accs, f1s, thresholds = tune_threshold(preds, labels)
        auc = roc_auc_score(labels, preds)

        print(f"✅ Chunk {chunk_id} 평가 완료 | Best Threshold = {best_t:.2f} | AUC: {auc:.4f} | 실패: {failed}")
        thresholds_per_chunk.append((chunk_id, best_t, auc))

        # 시각화
        plt.figure(figsize=(8, 4))
        plt.plot(thresholds, accs, label='Accuracy')
        plt.plot(thresholds, f1s, label='F1-score')
        plt.axvline(best_t, color='r', linestyle='--', label=f'Best T = {best_t:.2f}')
        plt.title(f'Threshold Tuning - Chunk {chunk_id}')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        all_preds.extend(preds)
        all_labels.extend(labels)

    # 전체 요약
    print(f"\n📊 전체 평가 쌍 수: {len(all_preds)}")
    print(f"🎯 전체 Accuracy (th=0.5): {accuracy_score(all_labels, [1 if s > 0.5 else 0 for s in all_preds]):.4f}")
    print(f"📈 전체 AUC: {roc_auc_score(all_labels, all_preds):.4f}")

    return thresholds_per_chunk

# -------------------------------------------------
# 📌 실행
# -------------------------------------------------
if __name__ == "__main__":
    evaluate_and_tune(csv_path, model_path, chunk_size=10000, start_chunk=0)
