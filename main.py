import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from skimage.feature import hog

# ----------------------------
# 설정
# ----------------------------
image_shape = (128, 128)
handcrafted_dim = 9
threshold = 0.5  # 조정 가능

# ----------------------------
# 유클리드 거리 함수
# ----------------------------
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

# ----------------------------
# 모델 로드
# ----------------------------
model_path = "/Users/chanyoungko/Desktop/HandWriting/model/siamese_handcrafted_model.h5"
model = load_model(model_path, custom_objects={'euclidean_distance': euclidean_distance})

# ----------------------------
# 이미지 전처리 및 특징 추출
# ----------------------------
def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {path}")
    img = cv2.resize(img, (image_shape[1], image_shape[0]))
    return img.astype(np.float32)

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

# ----------------------------
# 비교 및 시각화 함수
# ----------------------------
def compare_images(path1, path2):
    img1 = preprocess_image(path1)
    img2 = preprocess_image(path2)
    hand1 = extract_handcrafted_features(img1)
    hand2 = extract_handcrafted_features(img2)

    img1_input = np.expand_dims(img1 / 255.0, axis=(0, -1))
    img2_input = np.expand_dims(img2 / 255.0, axis=(0, -1))
    hand1_input = np.expand_dims(hand1, axis=0)
    hand2_input = np.expand_dims(hand2, axis=0)

    distance = model.predict([img1_input, hand1_input, img2_input, hand2_input])[0][0]
    similarity = 1 / (1 + distance)

    # --- 출력 ---
    print("📏 유클리드 거리:", f"{distance:.4f}")
    print("🔍 유사도 (1/(1+d)):", f"{similarity:.4f}")
    print("🎯 threshold 기준값:", threshold)
    if distance < threshold:
        result = "✅ 같은 사람일 가능성이 높습니다."
    else:
        result = "❌ 다른 사람일 가능성이 높습니다."
    print("📢 판별 결과:", result)

    # --- 시각화 ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    axes[0].imshow(img1, cmap='gray')
    axes[0].set_title("Image 1")
    axes[0].axis('off')

    axes[1].imshow(img2, cmap='gray')
    axes[1].set_title("Image 2")
    axes[1].axis('off')

    plt.suptitle(f"📏 Distance: {distance:.4f} | 🔍 Similarity: {similarity:.4f}\n🎯 Threshold: {threshold} → {result}", fontsize=12)
    plt.tight_layout()
    plt.show()

# ----------------------------
# 실행 예시
# ----------------------------
if __name__ == "__main__":
    path1 = "/Users/chanyoungko/Desktop/HandWriting/reference_samples/스크린샷 2025-05-13 오후 12.53.49.png"
    path2 = "/Users/chanyoungko/Desktop/HandWriting/test_samples/스크린샷 2025-05-13 오후 12.54.32.png"
    compare_images(path1, path2)
