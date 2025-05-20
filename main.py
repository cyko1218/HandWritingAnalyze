import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import glob


# ============ 필요한 함수/클래스 정의 ============
# 1. L1DistanceLayer 정의
class L1DistanceLayer(tf.keras.layers.Layer):
    """L1 거리를 계산하는 커스텀 레이어"""

    def __init__(self, **kwargs):
        super(L1DistanceLayer, self).__init__(**kwargs)

    def call(self, inputs):
        """두 입력 텐서 간의 L1 거리 계산"""
        return tf.abs(inputs[0] - inputs[1])

    def compute_output_shape(self, input_shape):
        """출력 형태 계산"""
        return input_shape[0]

    def get_config(self):
        """레이어 설정 반환"""
        config = super(L1DistanceLayer, self).get_config()
        return config


# 2. contrastive_loss 함수 정의
def contrastive_loss(y_true, y_pred, margin=1.0):
    """
    시암 네트워크를 위한 대조 손실 함수
    y_true: 1이면 같은 클래스, 0이면 다른 클래스
    y_pred: 두 입력 사이의 유클리드 거리
    margin: 다른 클래스 샘플 간 최소 거리
    """
    y_true = tf.cast(y_true, tf.float32)
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)


# 3. 이미지 전처리 함수
def preprocess_image(image_path, target_height=64, target_width=512):
    """필기체 이미지 전처리 함수 (가로로 긴 이미지 처리)"""
    try:
        # 이미지 로드
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")

        # 그레이스케일 변환
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # 원본 이미지 유지 (수동 특징 추출용)
        original_gray = gray.copy()

        # 이진화
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 노이즈 제거
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 종횡비 유지를 위한 처리
        h, w = binary.shape
        aspect = w / h

        # 종횡비에 따른 리사이징 조정
        if aspect >= target_width / target_height:
            # 너비가 더 길면 너비에 맞추고 높이 조정
            new_width = target_width
            new_height = int(target_width / aspect)
            # 높이가 너무 작으면 최소 높이 보장
            if new_height < target_height / 2:
                new_height = target_height // 2
        else:
            # 높이가 더 길면 높이에 맞추고 너비 조정
            new_height = target_height
            new_width = int(target_height * aspect)
            # 너비가 너무 작으면 최소 너비 보장
            if new_width < target_width / 2:
                new_width = target_width // 2

        # 리사이징
        resized = cv2.resize(binary, (new_width, new_height))

        # 고정 크기 캔버스 생성 (패딩 적용)
        canvas = np.zeros((target_height, target_width), dtype=np.uint8)

        # 중앙 배치
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2

        # 이미지 복사
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized

        # 정규화
        normalized = canvas.astype(np.float32) / 255.0

        # 차원 확장 (H, W) -> (H, W, 1) - 명시적으로 형태 확인
        if len(normalized.shape) == 2:  # 2D 이미지인 경우
            expanded = np.expand_dims(normalized, axis=-1)
        else:
            expanded = normalized

        # 형태 확인 및 강제 변환
        if expanded.shape != (target_height, target_width, 1):
            expanded = np.reshape(expanded, (target_height, target_width, 1))

        # 수동 특징 추출
        handcrafted_features = extract_handcrafted_features(original_gray, binary)

        return expanded, handcrafted_features

    except Exception as e:
        print(f"이미지 전처리 오류 ({image_path}): {e}")
        return None, None


# 4. 수동 특징 추출 함수
def extract_handcrafted_features(gray_img, binary_img=None):
    """필기체 이미지에서 수동 특징 추출"""
    features = []
    HANDCRAFTED_FEATURES_DIM = 12  # 수동 추출 특징 차원

    # 이진화 이미지가 제공되지 않은 경우
    if binary_img is None:
        _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 1. 픽셀 밀도 (필압 관련)
    pixel_density = np.sum(binary_img > 0) / binary_img.size
    features.append(pixel_density)

    # 2. 윤곽선 추출
    contours, _ = cv2.findContours(binary_img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 3. 기울기 분석
    angles = []
    for contour in contours:
        if len(contour) > 5:  # 타원 피팅에 필요한 최소 포인트
            try:
                ellipse = cv2.fitEllipse(contour)
                angle = ellipse[2]
                # 각도 표준화 (0-180)
                if angle > 90:
                    angle = angle - 180
                angles.append(angle)
            except:
                pass

    # 평균 기울기
    if angles:
        mean_angle = np.mean(angles)
        std_angle = np.std(angles)
    else:
        mean_angle = 0
        std_angle = 0

    features.append(mean_angle / 90)  # 정규화
    features.append(std_angle / 45)  # 정규화

    # 4. 크기 및 비율 분석
    if contours:
        heights = []
        widths = []
        areas = []
        aspect_ratios = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if area > 20:  # 노이즈 필터링
                heights.append(h)
                widths.append(w)
                areas.append(area)
                aspect_ratios.append(w / h if h > 0 else 0)

        if heights and widths:
            mean_height = np.mean(heights)
            std_height = np.std(heights)
            mean_width = np.mean(widths)
            std_width = np.std(widths)
            mean_area = np.mean(areas)
            mean_aspect = np.mean(aspect_ratios)
            std_aspect = np.std(aspect_ratios)
        else:
            mean_height = 0
            std_height = 0
            mean_width = 0
            std_width = 0
            mean_area = 0
            mean_aspect = 0
            std_aspect = 0
    else:
        mean_height = 0
        std_height = 0
        mean_width = 0
        std_width = 0
        mean_area = 0
        mean_aspect = 0
        std_aspect = 0

    features.append(mean_height / 100)  # 정규화
    features.append(std_height / 50)  # 정규화
    features.append(mean_width / 100)  # 정규화
    features.append(std_width / 50)  # 정규화
    features.append(mean_area / 1000)  # 정규화
    features.append(mean_aspect)
    features.append(std_aspect)

    # 특징을 최대 HANDCRAFTED_FEATURES_DIM 차원으로 제한
    features = features[:HANDCRAFTED_FEATURES_DIM]

    # 부족한 차원은 0으로 채움
    if len(features) < HANDCRAFTED_FEATURES_DIM:
        features.extend([0] * (HANDCRAFTED_FEATURES_DIM - len(features)))

    return np.array(features, dtype=np.float32)


# ============ 메인 코드 ============
def get_similarity(model_path, image1_path, image2_path):
    """
    두 이미지의 유사도를 계산하고 변수에 저장

    Args:
        model_path: 모델 파일 경로
        image1_path: 첫 번째 이미지 경로
        image2_path: 두 번째 이미지 경로

    Returns:
        float: similarity 값 (0~1)
    """
    # 커스텀 객체 정의
    custom_objects = {
        'L1DistanceLayer': L1DistanceLayer,
        'contrastive_loss': contrastive_loss
    }

    # 모델 로드
    print(f"모델 로드 중: {model_path}")
    model = load_model(model_path, custom_objects=custom_objects)
    print("모델 로드 완료")

    # 이미지 전처리
    print(f"이미지 전처리 중: {image1_path}")
    img1_result = preprocess_image(image1_path)
    if img1_result[0] is None:
        print(f"첫 번째 이미지 처리 실패: {image1_path}")
        return None

    print(f"이미지 전처리 중: {image2_path}")
    img2_result = preprocess_image(image2_path)
    if img2_result[0] is None:
        print(f"두 번째 이미지 처리 실패: {image2_path}")
        return None

    img1, hand1 = img1_result
    img2, hand2 = img2_result

    # 배치 형태로 변환
    print("예측 준비 중...")
    img1_batch = np.expand_dims(img1, axis=0)
    hand1_batch = np.expand_dims(hand1, axis=0)
    img2_batch = np.expand_dims(img2, axis=0)
    hand2_batch = np.expand_dims(hand2, axis=0)

    # 🔥 similarity 값 계산하고 변수에 저장
    print("유사도 계산 중...")
    similarity = model.predict([img1_batch, hand1_batch, img2_batch, hand2_batch])[0][0]

    # similarity 값 출력
    print("=" * 50)
    print(f"Similarity: {similarity}")
    print(f"Similarity: {similarity:.4f}")  # 소수점 4자리까지
    print("=" * 50)

    return similarity


# ============ 새로운 함수: 참조 폴더 변경 ============
def compare_with_custom_references(model_path, test_image_path, reference_folder):
    """
    테스트 이미지와 참조 폴더 내의 모든 이미지를 비교

    Args:
        model_path: 모델 파일 경로
        test_image_path: 테스트 이미지 경로
        reference_folder: 참조 이미지가 있는 폴더 경로
    """
    # 참조 폴더에서 이미지 파일 목록 가져오기
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    reference_files = []

    for ext in image_extensions:
        reference_files.extend(glob.glob(os.path.join(reference_folder, ext)))
        reference_files.extend(glob.glob(os.path.join(reference_folder, ext.upper())))

    # 파일이 없으면 종료
    if not reference_files:
        print(f"참조 폴더에 이미지 파일이 없습니다: {reference_folder}")
        return

    # 각 참조 이미지와 비교
    results = []
    for ref_path in reference_files:
        # 유사도 계산
        similarity_score = get_similarity(model_path, ref_path, test_image_path)

        if similarity_score is not None:
            # 결과 저장
            results.append({
                'reference_path': ref_path,
                'reference_name': os.path.basename(ref_path),
                'similarity': similarity_score
            })

            # 저장된 similarity 값 사용
            print(f"저장된 similarity: {similarity_score}")
            print(f"같은 저자인가? {'YES' if similarity_score >= 0.5 else 'NO'}")

            # similarity 값으로 다른 작업들
            if similarity_score > 0.8:
                print("매우 높은 유사도!")
            elif similarity_score > 0.6:
                print("높은 유사도")
            elif similarity_score > 0.4:
                print("중간 유사도")
            else:
                print("낮은 유사도")

            print("\n" + "-" * 50 + "\n")

    # 결과 요약
    if results:
        # 가장 유사한 이미지 찾기
        max_similarity = max(results, key=lambda x: x['similarity'])

        print("\n" + "=" * 50)
        print(f"가장 유사한 참조 이미지: {max_similarity['reference_name']}")
        print(f"유사도: {max_similarity['similarity']}")
        print("=" * 50)

    return results


# 간단한 사용 예시
if __name__ == "__main__":
    # 모델과 이미지 경로 설정
    model_path = "handwriting_hybrid_model_1.keras"  # 모델 경로 수정하세요

    # 👉 참조 폴더 변경 - 이 부분만 수정했습니다!
    reference_folder = "/Users/chanyoungko/Desktop/HandWriting/custom_references"  # 참조 폴더 경로
    test_image_path = "/reference_samples/img.png"  # 테스트 이미지 경로

    # 사용자 입력 처리
    print("\n" + "=" * 50)
    print("필기체 비교 시스템")
    print("=" * 50)
    import os
    import cv2
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras import layers


    # ============ 커스텀 레이어 ============
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


    # ============ 손실 함수 ============
    def contrastive_loss(y_true, y_pred, margin=1.0):
        y_true = tf.cast(y_true, tf.float32)
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(margin - y_pred, 0))
        return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)


    # ============ 수동 특징 추출 ============
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


    # ============ 이미지 전처리 ============
    def preprocess_image(image_path, target_height=64, target_width=512):
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")

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


    # ============ 유사도 계산 ============
    def get_similarity(model, image1_path, image2_path):
        img1_result = preprocess_image(image1_path)
        img2_result = preprocess_image(image2_path)

        if img1_result[0] is None or img2_result[0] is None:
            return None

        img1, hand1 = img1_result
        img2, hand2 = img2_result

        img1_batch = np.expand_dims(img1, axis=0)
        hand1_batch = np.expand_dims(hand1, axis=0)
        img2_batch = np.expand_dims(img2, axis=0)
        hand2_batch = np.expand_dims(hand2, axis=0)

        similarity = model.predict([img1_batch, hand1_batch, img2_batch, hand2_batch])[0][0]

        print("=" * 50)
        print(f"[비교] {os.path.basename(image1_path)} vs {os.path.basename(image2_path)}")
        print(f"Similarity: {similarity:.4f}")
        print("=" * 50)
        return similarity


    def create_result(results):
        if not results:
            print("❌ 비교할 결과 없음")
            exit(1)

        best_result = results[0]

        #return AnalyzeResponse(best_result['avg_similarity'], best_result['avg_pressure'], best_result['avg_slant'], "")

    # ============ 메인 실행 ============
    if __name__ == "__main__":
        model_path = "handwriting_hybrid_model_1.keras"
        reference_folder = "/Users/chanyoungko/Desktop/HandWriting/reference_samples"
        test_image_path = "/Users/chanyoungko/Desktop/HandWriting/test_samples/img.png"

        custom_objects = {
            'L1DistanceLayer': L1DistanceLayer,
            'contrastive_loss': contrastive_loss
        }

        print(f"모델 로드 중: {model_path}")
        model = load_model(model_path, custom_objects=custom_objects)
        print("모델 로드 완료")

        similarity_scores = []

        for filename in os.listdir(reference_folder):
            ref_path = os.path.join(reference_folder, filename)
            if os.path.isfile(ref_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                score = get_similarity(model, ref_path, test_image_path)
                if score is not None:
                    similarity_scores.append(score)

        if similarity_scores:
            avg_score = np.mean(similarity_scores)
            print("\n" + "#" * 50)
            print(f"🔍 전체 평균 유사도: {avg_score:.4f}")
            print(f"✔️ 비교한 이미지 수: {len(similarity_scores)}")
            print("#" * 50)
        else:
            print("❌ 유사도 계산에 실패했습니다.")

        if similarity_scores:
            avg_score = np.mean(similarity_scores)
            print("\n" + "#" * 50)
            print(f"🔍 전체 평균 유사도: {avg_score:.4f}")
            print(f"✔️ 비교한 이미지 수: {len(similarity_scores)}")

            # Threshold 비교
            threshold = 0.5
            print("#" * 50)
            if avg_score >= threshold:
                print(f"✅ 판별 결과: 같은 사람입니다 (유사도 ≥ {threshold})")
            else:
                print(f"❌ 판별 결과: 다른 사람입니다 (유사도 < {threshold})")
            print("#" * 50)
        else:
            print("❌ 유사도 계산에 실패했습니다.")




    # 또는 개별 이미지 비교를 원하는 경우 (원래 코드)
    # image1_path = "/path/to/reference_image.png"
    # similarity_score = get_similarity(model_path, image1_path, test_image_path)
    #
    # if similarity_score is not None:
    #     # 저장된 similarity 값 사용
    #     print(f"저장된 similarity: {similarity_score}")
    #     print(f"같은 저자인가? {'YES' if similarity_score >= 0.5 else 'NO'}")
    #
    #     # similarity 값으로 다른 작업들
    #     if similarity_score > 0.8:
    #         print("매우 높은 유사도!")
    #     elif similarity_score > 0.6:
    #         print("높은 유사도")
    #     elif similarity_score > 0.4:
    #         print("중간 유사도")
    #     else:
    #         print("낮은 유사도")