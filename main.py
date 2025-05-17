# ✅ 줄 단위 필기체 비교 및 시각화 시스템 (keras 모델 기반)

import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tensorflow.keras.models import load_model
from skimage.feature import hog
from tensorflow.keras import backend as K

# ----------------------------
# 설정
# ----------------------------
model_path = "siamese_handcrafted_model_memory.keras"
reference_dir = "reference_samples"
test_dir = "test_samples"

image_shape = (88, 765)
handcrafted_dim = 9

# ----------------------------
# 유사도 변환 함수
# ----------------------------
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def distance_to_similarity(distance, factor=5):
    return math.exp(-distance * factor)

# ----------------------------
# handcrafted 특성 추출
# ----------------------------
def extract_handcrafted_features(img):
    img_uint8 = img.astype(np.uint8)
    _, binary = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    features = [
        np.mean(img) / 255.0,
        np.std(img) / 255.0
    ]
    hog_feat = hog(binary, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, feature_vector=True)
    hog_feat = hog_feat[:7] if len(hog_feat) >= 7 else np.pad(hog_feat, (0, 7 - len(hog_feat)))
    features.extend(hog_feat)
    return np.array(features[:9], dtype=np.float32)

# ----------------------------
# 이미지 줄 추출 (수평 프로젝션)
# ----------------------------
def extract_lines_from_image(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h_proj = np.sum(binary, axis=1)

    lines = []
    in_line = False
    start = 0
    for i, val in enumerate(h_proj):
        if val > 0 and not in_line:
            start = i
            in_line = True
        elif val == 0 and in_line:
            end = i
            if end - start > 5:
                lines.append(img[max(0, start - 5):min(img.shape[0], end + 5), :])
            in_line = False
    return lines

# ----------------------------
# 줄 단위 유사도 비교 함수
# ----------------------------
def compare_lines_with_model(model, test_lines, ref_lines):
    sim_matrix = np.zeros((len(test_lines), len(ref_lines)))
    for i, test in enumerate(test_lines):
        test_img = cv2.resize(test, image_shape).astype(np.float32) / 255.0
        test_hand = extract_handcrafted_features(test * 255.0)
        test_input_img = np.expand_dims(test_img[..., np.newaxis], axis=0)
        test_input_hand = np.expand_dims(test_hand, axis=0)

        for j, ref in enumerate(ref_lines):
            ref_img = cv2.resize(ref, image_shape).astype(np.float32) / 255.0
            ref_hand = extract_handcrafted_features(ref * 255.0)
            ref_input_img = np.expand_dims(ref_img[..., np.newaxis], axis=0)
            ref_input_hand = np.expand_dims(ref_hand, axis=0)

            distance = model.predict([test_input_img, test_input_hand, ref_input_img, ref_input_hand], verbose=0)[0][0]
            sim_matrix[i, j] = distance_to_similarity(distance)
    return sim_matrix

# ----------------------------
# 시각화 함수
# ----------------------------
def visualize_similarity(test_img, ref_img, similarity_matrix, test_lines, ref_lines):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(test_img, cmap='gray')
    plt.title("Test Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(ref_img, cmap='gray')
    plt.title("Reference Image")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(similarity_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Similarity')
    plt.title("Line Similarity Matrix")
    plt.xlabel("Reference Lines")
    plt.ylabel("Test Lines")
    plt.tight_layout()
    plt.show()

# ----------------------------
# 실행
# ----------------------------
def run_compare():
    # 모델 로드
    model = load_model(model_path, custom_objects={"euclidean_distance": euclidean_distance})

    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(('png', 'jpg', 'jpeg'))]
    ref_files = [os.path.join(reference_dir, f) for f in os.listdir(reference_dir) if f.endswith(('png', 'jpg', 'jpeg'))]

    test_path = test_files[0]
    test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    test_lines = extract_lines_from_image(test_img)

    for ref_path in ref_files:
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        ref_lines = extract_lines_from_image(ref_img)

        sim_matrix = compare_lines_with_model(model, test_lines, ref_lines)
        avg_sim = np.mean(sim_matrix)
        print(f"\n📄 참조 문서: {os.path.basename(ref_path)}")
        print(f"평균 유사도: {avg_sim:.4f}")
        visualize_similarity(test_img, ref_img, sim_matrix, test_lines, ref_lines)

if __name__ == "__main__":
    run_compare()
