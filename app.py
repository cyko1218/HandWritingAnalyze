import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import glob

def is_handwriting_image(image, pixel_threshold_ratio=0.01, min_contours=5):
    """
    이미지에 글씨가 포함되어 있는지 판단
    - 픽셀 기준: 전체 픽셀 중 어두운 영역의 비율
    - 컨투어 기준: 윤곽선(획) 개수
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 이진화 및 닫기 연산 (노이즈 제거)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

    # 1. 픽셀 비율 검사
    pixel_ratio = np.sum(binary > 0) / binary.size
    if pixel_ratio < pixel_threshold_ratio:
        return False  # 너무 비어 있음

    # 2. 컨투어 개수 검사
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < min_contours:
        return False

    return True


# ========================= 커스텀 레이어 =========================
class L1DistanceLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(L1DistanceLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.abs(inputs[0] - inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(L1DistanceLayer, self).get_config()
        return config

# ========================= Contrastive Loss =========================
def contrastive_loss(y_true, y_pred, margin=1.0):
    y_true = tf.cast(y_true, tf.float32)
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

# ========================= 수동 특징 추출 =========================
def extract_handcrafted_features(gray_img, binary_img=None):
    features = []
    HANDCRAFTED_FEATURES_DIM = 12

    if binary_img is None:
        _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    pixel_density = np.sum(binary_img > 0) / binary_img.size
    features.append(pixel_density)

    contours, _ = cv2.findContours(binary_img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    angles = []
    for contour in contours:
        if len(contour) > 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                angle = ellipse[2]
                if angle > 90:
                    angle -= 180
                angles.append(angle)
            except:
                pass

    mean_angle = np.mean(angles) if angles else 0
    std_angle = np.std(angles) if angles else 0
    features.append(mean_angle / 90)
    features.append(std_angle / 45)

    heights, widths, areas, aspect_ratios = [], [], [], []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 20:
            heights.append(h)
            widths.append(w)
            areas.append(area)
            aspect_ratios.append(w / h if h > 0 else 0)

    if heights and widths:
        features.extend([
            np.mean(heights) / 100,
            np.std(heights) / 50,
            np.mean(widths) / 100,
            np.std(widths) / 50,
            np.mean(areas) / 1000,
            np.mean(aspect_ratios),
            np.std(aspect_ratios)
        ])
    else:
        features.extend([0] * 7)

    features = features[:HANDCRAFTED_FEATURES_DIM]
    features.extend([0] * (HANDCRAFTED_FEATURES_DIM - len(features)))
    return np.array(features, dtype=np.float32)

# ========================= 이미지 전처리 =========================
def preprocess_image(image_path, target_height=64, target_width=512):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")

        # 👇 글씨 유무 검사 추가
        if not is_handwriting_image(img):
            raise ValueError("⚠️ 글씨가 없는 이미지입니다.")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        original_gray = gray.copy()

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

        h, w = binary.shape
        aspect = w / h
        if aspect >= target_width / target_height:
            new_width = target_width
            new_height = max(int(target_width / aspect), target_height // 2)
        else:
            new_height = target_height
            new_width = max(int(target_height * aspect), target_width // 2)

        resized = cv2.resize(binary, (new_width, new_height))
        canvas = np.zeros((target_height, target_width), dtype=np.uint8)
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized

        normalized = canvas.astype(np.float32) / 255.0
        expanded = np.expand_dims(normalized, axis=-1)
        if expanded.shape != (target_height, target_width, 1):
            expanded = np.reshape(expanded, (target_height, target_width, 1))

        handcrafted_features = extract_handcrafted_features(original_gray, binary)
        return expanded, handcrafted_features

    except Exception as e:
        print(f"이미지 전처리 오류 ({image_path}): {e}")
        return None, None


# ========================= 유사도 계산 =========================
def get_similarity(model, image1_path, image2_path):
    img1_result = preprocess_image(image1_path)
    img2_result = preprocess_image(image2_path)

    if img1_result[0] is None or img2_result[0] is None:
        return None, None, None

    img1, hand1 = img1_result
    img2, hand2 = img2_result

    img1_batch = np.expand_dims(img1, axis=0)
    hand1_batch = np.expand_dims(hand1, axis=0)
    img2_batch = np.expand_dims(img2, axis=0)
    hand2_batch = np.expand_dims(hand2, axis=0)

    similarity = model.predict([img1_batch, hand1_batch, img2_batch, hand2_batch])[0][0]

    pressure = (hand1[0] + hand2[0]) / 2
    slant = (hand1[1] + hand2[1]) / 2

    return similarity, pressure, slant

# ========================= 결과 생성 =========================
def create_result(results, avg_score):
    if not results:
        print("❌ 비교할 결과 없음")
        return None

    best_result = results[0]
    avg_pressure = best_result.get('pressure', 0.0)
    avg_slant = best_result.get('slant', 0.0)

    print("\n" + "=" * 50)
    print("📝 최종 결과 요약")
    print(f"📌 평균 유사도: {avg_score*100:.4f}%")
    print(f"📌 평균 필압: {avg_pressure:.4f}")
    print(f"📌 평균 기울기: {avg_slant:.4f}")
    print("=" * 50)

    return {
        'similarity': avg_score,
        'pressure': avg_pressure,
        'slant': avg_slant
    }

# ========================= 전체 실행 =========================
if __name__ == "__main__":
    model_path = "handwriting_hybrid_model_1.keras"
    reference_folder = "/Users/chanyoungko/Desktop/HandWriting/reference_samples"
    test_image_path = "/Users/chanyoungko/Desktop/HandWriting/test_samples/img.png"

    print(f"모델 로드 중: {model_path}")
    custom_objects = {'L1DistanceLayer': L1DistanceLayer, 'contrastive_loss': contrastive_loss}
    model = load_model(model_path, custom_objects=custom_objects)
    print("모델 로드 완료")

    similarity_scores = []

    for filename in os.listdir(reference_folder):
        ref_path = os.path.join(reference_folder, filename)
        if os.path.isfile(ref_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            similarity, pressure, slant = get_similarity(model, ref_path, test_image_path)
            if similarity is not None:
                similarity_scores.append({
                    'reference': filename,
                    'similarity': similarity,
                    'pressure': pressure,
                    'slant': slant
                })

    similarity_scores.sort(key=lambda x: x['similarity'], reverse=True)

    if similarity_scores:
        avg_score = np.mean([item['similarity'] for item in similarity_scores])
        print("\n" + "#" * 50)
        print(f"🔍 전체 평균 유사도: {avg_score:.4f}")
        print(f"✔️ 비교한 이미지 수: {len(similarity_scores)}")
        print("#" * 50)

        threshold = 0.5
        print("#" * 50)
        if avg_score >= threshold:
            print(f"✅ 판별 결과: 같은 사람입니다 (유사도 ≥ {threshold})")
        else:
            print(f"❌ 판별 결과: 다른 사람입니다 (유사도 < {threshold})")
        print("#" * 50)

        summary = create_result(similarity_scores, avg_score)

    else:
        print("❌ 유사도 계산에 실패했습니다.")
