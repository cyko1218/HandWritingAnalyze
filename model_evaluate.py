import os
import cv2
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score

from model_train import build_siamese_model, preprocess_image, extract_handcrafted_features, euclidean_distance
from tensorflow.keras.models import Model

# ✅ 유사도 변환 함수
def distance_to_similarity(distance, factor=5):
    return np.exp(-distance * factor)

# ✅ 평가 함수
def evaluate_model_from_csv(csv_path, weights_path, threshold=0.5):
    df = pd.read_csv(csv_path)

    # 모델 구조 불러오기
    model = build_siamese_model()
    model.load_weights(weights_path)

    predictions = []
    labels = []
    failed = 0
    start_time = time.time()

    print(f"🧪 총 {len(df)}쌍 평가 시작...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="🔍 Evaluating"):
        path1 = row['image_path_1']
        path2 = row['image_path_2']
        label = row['label']

        img1 = preprocess_image(path1)
        img2 = preprocess_image(path2)

        if img1 is None or img2 is None:
            print(f"⚠️ 이미지 로드 실패: {path1} 또는 {path2}")
            failed += 1
            continue

        hand1 = extract_handcrafted_features(img1)
        hand2 = extract_handcrafted_features(img2)

        # 모델 입력 형태로 변환
        img1_input = np.expand_dims(img1 / 255.0, axis=(0, -1))
        img2_input = np.expand_dims(img2 / 255.0, axis=(0, -1))
        hand1_input = np.expand_dims(hand1, axis=0)
        hand2_input = np.expand_dims(hand2, axis=0)

        # 예측 수행
        distance = model.predict([img1_input, hand1_input, img2_input, hand2_input], verbose=0)[0][0]
        similarity = distance_to_similarity(distance)

        predictions.append(similarity)
        labels.append(label)

    elapsed = time.time() - start_time
    binary_preds = [1 if s > threshold else 0 for s in predictions]

    acc = accuracy_score(labels, binary_preds)
    auc = roc_auc_score(labels, predictions)

    print(f"\n✅ 평가 완료: {len(predictions)}쌍 성공, {failed}쌍 실패")
    print(f"🎯 Accuracy: {acc:.4f}, 📈 AUC: {auc:.4f}, ⏱️ Time: {elapsed:.2f} sec")

    return pd.DataFrame({
        'image_path_1': df['image_path_1'][:len(predictions)],
        'image_path_2': df['image_path_2'][:len(predictions)],
        'label': labels,
        'predicted_similarity': predictions,
        'predicted_label': binary_preds
    }), acc, auc

# ✅ 실행 예시
if __name__ == "__main__":
    csv_path = "/Users/chanyoungko/Desktop/HandWriting/handwriting_pairs.csv"
    weights_path = "/Users/chanyoungko/Desktop/HandWriting/model/siamese_handcrafted_model.h5"  # model.save_weights()로 저장한 파일

    result_df, acc, auc = evaluate_model_from_csv(csv_path, weights_path)
    result_df.to_csv("evaluation_result.csv", index=False)
