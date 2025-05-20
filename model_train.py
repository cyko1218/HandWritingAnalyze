import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras import backend as K
import time
import datetime

# Apple Silicon GPU 활성화
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"✅ GPU 활성화됨: {physical_devices}")
    else:
        print("⚠️ GPU를 찾을 수 없습니다. CPU로 실행됩니다.")
except Exception as e:
    print(f"⚠️ GPU 활성화 오류: {e}")
    print("CPU로 실행됩니다.")

# Mixed Precision 활성화 (MacBook M 시리즈 성능 향상)
try:
    from tensorflow.keras.mixed_precision import set_global_policy

    set_global_policy('mixed_float16')
    print("✅ Mixed Precision (FP16) 활성화됨")
except Exception as e:
    print(f"⚠️ Mixed Precision 활성화 실패: {e}")

# 설정
IMAGE_HEIGHT = 64  # 높이 고정
IMAGE_WIDTH = 512  # 너비 고정 (필기체 이미지에 맞게 조정)
BATCH_SIZE = 16  # MacBook 메모리에 맞게 조정
EPOCHS = 10
LEARNING_RATE = 1e-4
HANDCRAFTED_FEATURES_DIM = 12  # 수동 추출 특징 차원
CSV_PATH = "/Users/chanyoungko/Desktop/HandWriting/handwriting_pairs_shuffled.csv"  # CSV 파일 경로 지정
MODEL_PATH = "handwriting_hybrid_model_1.keras"  # 모델 저장 경로


# 대조 손실 함수 (Contrastive Loss)
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


# 이미지 전처리 함수
def preprocess_image(image_path, target_height=IMAGE_HEIGHT, target_width=IMAGE_WIDTH):
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


# 수동 특징 추출 함수
def extract_handcrafted_features(gray_img, binary_img=None):
    """필기체 이미지에서 수동 특징 추출"""
    features = []

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

    # 5. 필기 모양 통계
    # 이미지 크기 정규화 (여러 이미지 간 일관성을 위해)
    resized_binary = cv2.resize(binary_img, (100, 50))

    # 행별 필기 채우기 (각 행에서 필기가 차지하는 비율)
    row_fill = np.mean(resized_binary > 0, axis=1)
    mean_row_fill = np.mean(row_fill)
    std_row_fill = np.std(row_fill)

    # 열별 필기 채우기 (각 열에서 필기가 차지하는 비율)
    col_fill = np.mean(resized_binary > 0, axis=0)
    mean_col_fill = np.mean(col_fill)
    std_col_fill = np.std(col_fill)

    features.append(mean_row_fill)
    features.append(std_row_fill)
    features.append(mean_col_fill)
    features.append(std_col_fill)

    # 특징을 최대 HANDCRAFTED_FEATURES_DIM 차원으로 제한
    features = features[:HANDCRAFTED_FEATURES_DIM]

    # 부족한 차원은 0으로 채움
    if len(features) < HANDCRAFTED_FEATURES_DIM:
        features.extend([0] * (HANDCRAFTED_FEATURES_DIM - len(features)))

    return np.array(features, dtype=np.float32)


# 데이터 증강 함수
def augment_image(img, handcrafted_features):
    """필기체 이미지 증강 함수"""
    # 입력 이미지 형태 확인
    if img.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 1):
        img = np.reshape(img, (IMAGE_HEIGHT, IMAGE_WIDTH, 1))

    h, w, c = img.shape

    # 랜덤 시프트 (필기체 위치 변화)
    max_shift_x = w // 20
    max_shift_y = h // 20
    shift_x = np.random.randint(-max_shift_x, max_shift_x + 1)
    shift_y = np.random.randint(-max_shift_y, max_shift_y + 1)

    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted = cv2.warpAffine(img, M, (w, h), borderValue=0)

    # 약간의 회전 (±3도, 필기 기울기 변화)
    angle = np.random.uniform(-3, 3)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    rotated = cv2.warpAffine(shifted, M, (w, h), borderValue=0)

    # 약간의 크기 변화 (95-105%, 필기 크기 변화)
    scale = np.random.uniform(0.95, 1.05)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
    scaled = cv2.warpAffine(rotated, M, (w, h), borderValue=0)

    # 밝기 변화
    brightness = np.random.uniform(0.9, 1.1)
    brightened = np.clip(scaled * brightness, 0, 1)

    # 수동 특징도 약간 변화 (노이즈 추가)
    noise_scale = 0.05  # 노이즈 스케일
    noisy_features = handcrafted_features + np.random.normal(0, noise_scale, handcrafted_features.shape)

    # 최종 형태 확인 및 조정
    if brightened.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 1):
        brightened = np.reshape(brightened, (IMAGE_HEIGHT, IMAGE_WIDTH, 1))

    return brightened, noisy_features


# CNN 특징 추출 모듈
def cnn_feature_extractor(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1)):
    """필기체 이미지를 위한 CNN 특징 추출기"""
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # 글로벌 풀링
    x = layers.GlobalAveragePooling2D()(x)

    # 특징 벡터
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128)(x)

    model = models.Model(inputs, x, name="cnn_feature_extractor")
    return model


# 수동 특징 처리 모듈
def handcrafted_feature_processor(input_dim=HANDCRAFTED_FEATURES_DIM):
    """수동 특징 처리를 위한 신경망"""
    inputs = layers.Input(shape=(input_dim,))

    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)

    model = models.Model(inputs, x, name="handcrafted_processor")
    return model


# 1. 커스텀 레이어 클래스 정의 (클래스 정의 섹션에 추가)
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

# 하이브리드 시암 네트워크 모델 구축
def build_hybrid_siamese_model():
    """필기체 하이브리드 시암 네트워크"""
    # 입력 정의
    img_input_a = layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1), name="img_input_a")
    img_input_b = layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1), name="img_input_b")
    hand_input_a = layers.Input(shape=(HANDCRAFTED_FEATURES_DIM,), name="hand_input_a")
    hand_input_b = layers.Input(shape=(HANDCRAFTED_FEATURES_DIM,), name="hand_input_b")

    # 특징 추출기 정의
    cnn_extractor = cnn_feature_extractor()
    hand_processor = handcrafted_feature_processor()

    # 특징 추출
    cnn_a = cnn_extractor(img_input_a)
    cnn_b = cnn_extractor(img_input_b)
    hand_a = hand_processor(hand_input_a)
    hand_b = hand_processor(hand_input_b)

    # 특징 결합
    combined_a = layers.Concatenate()([cnn_a, hand_a])
    combined_b = layers.Concatenate()([cnn_b, hand_b])

    # 융합 레이어
    fused_a = layers.Dense(96, activation='relu')(combined_a)
    fused_a = layers.BatchNormalization()(fused_a)
    fused_b = layers.Dense(96, activation='relu')(combined_b)
    fused_b = layers.BatchNormalization()(fused_b)

    # ✅ Lambda 레이어 대신 커스텀 레이어 사용
    l1_distance = L1DistanceLayer(name="l1_distance")([fused_a, fused_b])

    # 유사도 분류
    x = layers.Dense(64, activation='relu')(l1_distance)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    similarity = layers.Dense(1, activation='sigmoid', name="similarity")(x)

    # 모델 정의
    model = models.Model(
        inputs=[img_input_a, hand_input_a, img_input_b, hand_input_b],
        outputs=similarity,
        name="hybrid_siamese_network"
    )

    return model, cnn_extractor, hand_processor


# 간소화된 훈련 모니터링 콜백
class HybridHandwritingDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, csv_path, batch_size=32, shuffle=True, augment=False):
        super().__init__()  # super() 호출 추가
        self.data = pd.read_csv(csv_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = np.arange(len(self.data))

        # 첫 번째 이미지 확인하여 열 이름 추론
        if 'image_path_1' in self.data.columns:
            self.img1_col = 'image_path_1'
            self.img2_col = 'image_path_2'
        elif 'img1' in self.data.columns:
            self.img1_col = 'img1'
            self.img2_col = 'img2'
        else:
            # 첫 번째와 두 번째 열이 이미지 경로라고 가정
            self.img1_col = self.data.columns[0]
            self.img2_col = self.data.columns[1]

        # 라벨 열 이름 확인
        if 'label' in self.data.columns:
            self.label_col = 'label'
        elif 'is_same' in self.data.columns:
            self.label_col = 'is_same'
        else:
            # 세 번째 열이 라벨이라고 가정
            self.label_col = self.data.columns[2]

        print(f"CSV 열 구조: 이미지1={self.img1_col}, 이미지2={self.img2_col}, 라벨={self.label_col}")
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        # 배치 인덱스 가져오기
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.data))
        indexes = self.indexes[batch_start:batch_end]
        batch_data = self.data.iloc[indexes]

        # 실제 배치 크기 계산
        actual_batch_size = len(batch_data)

        # 이미지 배치와 라벨 배치 초기화
        # 명시적인 크기의 배열 미리 할당
        img1_batch = np.zeros((actual_batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.float32)
        img2_batch = np.zeros((actual_batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.float32)
        hand1_batch = np.zeros((actual_batch_size, HANDCRAFTED_FEATURES_DIM), dtype=np.float32)
        hand2_batch = np.zeros((actual_batch_size, HANDCRAFTED_FEATURES_DIM), dtype=np.float32)
        label_batch = np.zeros((actual_batch_size,), dtype=np.float32)

        # 유효한 데이터 수 추적
        valid_count = 0

        # 배치 데이터 로드 및 전처리
        for i, (_, row) in enumerate(batch_data.iterrows()):
            if valid_count >= actual_batch_size:
                break

            try:
                img1_path = row[self.img1_col]
                img2_path = row[self.img2_col]
                label = row[self.label_col]

                # 이미지 전처리
                result1 = preprocess_image(img1_path)
                result2 = preprocess_image(img2_path)

                # 전처리 실패 시 스킵
                if result1[0] is None or result2[0] is None:
                    continue

                img1, hand1 = result1
                img2, hand2 = result2

                # 데이터 증강 (학습용 데이터셋이고 augment=True인 경우)
                if self.augment:
                    if np.random.random() < 0.5:  # 50% 확률로 증강
                        img1, hand1 = augment_image(img1, hand1)
                    if np.random.random() < 0.5:
                        img2, hand2 = augment_image(img2, hand2)

                # 배치 배열에 할당
                img1_batch[valid_count] = img1
                img2_batch[valid_count] = img2
                hand1_batch[valid_count] = hand1
                hand2_batch[valid_count] = hand2
                label_batch[valid_count] = label

                valid_count += 1

            except Exception as e:
                print(f"배치 처리 오류: {e}")
                continue

        if valid_count == 0:  # 유효한 샘플이 없는 경우
            # 더미 데이터 반환
            dummy_img = np.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.float32)
            dummy_hand = np.zeros((1, HANDCRAFTED_FEATURES_DIM), dtype=np.float32)

            return {
                "img_input_a": dummy_img,
                "hand_input_a": dummy_hand,
                "img_input_b": dummy_img,
                "hand_input_b": dummy_hand
            }, np.array([0], dtype=np.float32)

        # 실제 유효한 샘플 수에 맞게 배열 크기 조정
        if valid_count < actual_batch_size:
            img1_batch = img1_batch[:valid_count]
            img2_batch = img2_batch[:valid_count]
            hand1_batch = hand1_batch[:valid_count]
            hand2_batch = hand2_batch[:valid_count]
            label_batch = label_batch[:valid_count]

        # 딕셔너리로 반환
        return {
            "img_input_a": img1_batch,
            "hand_input_a": hand1_batch,
            "img_input_b": img2_batch,
            "hand_input_b": hand2_batch
        }, label_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


# 간소화된 모델 훈련 함수
def train_hybrid_model_simple(csv_path, model_path):
    """하이브리드 시암 네트워크 모델 훈련 (단순화된 버전)"""
    # 시작 시간 기록
    start_time = time.time()

    # 데이터 분할
    print("데이터 로드 및 분할 중...")
    df = pd.read_csv(csv_path)

    # 간단한 데이터 탐색
    print(f"전체 데이터 수: {len(df)}")

    # 훈련/검증 분할 (80/20)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df.iloc[:, 2])

    print(f"훈련 데이터: {len(train_df)}개")
    print(f"검증 데이터: {len(val_df)}개")

    # 임시 CSV 파일 생성
    train_csv = 'train_temp.csv'
    val_csv = 'val_temp.csv'

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    # 데이터 생성기 초기화
    print("데이터 생성기 초기화 중...")
    train_gen = HybridHandwritingDataGenerator(
        train_csv, batch_size=BATCH_SIZE, augment=True, shuffle=True
    )
    val_gen = HybridHandwritingDataGenerator(
        val_csv, batch_size=BATCH_SIZE, augment=False, shuffle=False
    )

    # 모델 구축
    print("하이브리드 시암 네트워크 모델 구축 중...")
    model, _, _ = build_hybrid_siamese_model()

    # 모델 컴파일
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # 모델 정보 출력
    print(f"모델 입력 크기: {IMAGE_HEIGHT}x{IMAGE_WIDTH}x1")
    print(f"수동 특징 차원: {HANDCRAFTED_FEATURES_DIM}")
    print(f"배치 크기: {BATCH_SIZE}")

    # 콜백 정의
    callbacks = [
        # 모델 체크포인트
        ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # 학습률 감소
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # 조기 중단
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # 간소화된 훈련 모니터
        SimpleTrainingMonitor()
    ]

    # 모델 학습
    print(f"\n모델 훈련 시작... (에폭: {EPOCHS})")
    # model.fit 호출 부분 수정
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        # workers=4 제거 - 더 이상 지원되지 않음
        verbose=1
    )

    # 임시 파일 정리
    os.remove(train_csv)
    os.remove(val_csv)

    # 총 훈련 시간 계산
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\n모델 저장 경로: {model_path}")
    print(f"총 훈련 시간: {int(hours)}시간 {int(minutes)}분 {int(seconds)}초")

    return model, history


# 필기체 감정 함수 (검증 시 사용)
def verify_handwriting(model, reference_samples, query_sample, threshold=0.5):
    """
    여러 참조 샘플과 쿼리 샘플 비교

    Args:
        model: 학습된 하이브리드 시암 모델
        reference_samples: 참조 필기체 이미지 경로 리스트 (5장 권장)
        query_sample: 검증할 필기체 이미지 경로
        threshold: 판별 임계값

    Returns:
        dict: 검증 결과 및 세부 정보
    """
    # 쿼리 이미지 전처리
    query_img, query_hand = preprocess_image(query_sample)
    if query_img is None:
        return {"error": "검증 이미지 처리 실패"}

    # 참조 이미지 전처리
    reference_imgs = []
    reference_hands = []
    valid_references = []

    for i, ref_path in enumerate(reference_samples):
        result = preprocess_image(ref_path)
        if result[0] is not None:
            ref_img, ref_hand = result
            reference_imgs.append(ref_img)
            reference_hands.append(ref_hand)
            valid_references.append(ref_path)

    if not reference_imgs:
        return {"error": "유효한 참조 이미지가 없습니다"}

    # 각 참조 이미지와 쿼리 이미지 비교
    similarities = []
    for i, (ref_img, ref_hand) in enumerate(zip(reference_imgs, reference_hands)):
        # 배치 형태로 변환
        img1_batch = np.expand_dims(ref_img, axis=0)
        hand1_batch = np.expand_dims(ref_hand, axis=0)
        img2_batch = np.expand_dims(query_img, axis=0)
        hand2_batch = np.expand_dims(query_hand, axis=0)

        # 유사도 예측
        similarity = model.predict([img1_batch, hand1_batch, img2_batch, hand2_batch])[0][0]
        similarities.append(float(similarity))

    # 결과 분석
    avg_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)
    is_same_author = avg_similarity >= threshold

    # 세부 결과
    details = []
    for i, (ref_path, sim) in enumerate(zip(valid_references, similarities)):
        details.append({
            "reference_index": i,
            "reference_path": ref_path,
            "similarity": sim,
            "match": sim >= threshold
        })

    # 최종 결과
    result = {
        "verified": bool(is_same_author),
        "average_similarity": float(avg_similarity),
        "similarity_std": float(std_similarity),
        "threshold": float(threshold),
        "confidence": float(1 - std_similarity) if is_same_author else float(std_similarity),
        "details": details
    }

    return result


# 최적 임계값 찾기 함수
def find_optimal_threshold(model, test_csv):
    """검증 데이터셋으로 최적 임계값 찾기"""
    df = pd.read_csv(test_csv)

    # 이미지 열과 라벨 열 확인
    if 'image_path_1' in df.columns:
        img1_col, img2_col = 'image_path_1', 'image_path_2'
    elif 'img1' in df.columns:
        img1_col, img2_col = 'img1', 'img2'
    else:
        img1_col, img2_col = df.columns[0], df.columns[1]

    if 'label' in df.columns:
        label_col = 'label'
    elif 'is_same' in df.columns:
        label_col = 'is_same'
    else:
        label_col = df.columns[2]

    # 유사도와 라벨 수집
    similarities = []
    true_labels = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="임계값 최적화 중"):
        img1_path = row[img1_col]
        img2_path = row[img2_col]
        true_label = row[label_col]

        # 이미지 전처리
        img1_data = preprocess_image(img1_path)
        img2_data = preprocess_image(img2_path)

        if img1_data[0] is None or img2_data[0] is None:
            continue

        img1, hand1 = img1_data
        img2, hand2 = img2_data

        # 배치 형태로 변환
        img1_batch = np.expand_dims(img1, axis=0)
        hand1_batch = np.expand_dims(hand1, axis=0)
        img2_batch = np.expand_dims(img2, axis=0)
        hand2_batch = np.expand_dims(hand2, axis=0)

        # 예측
        similarity = model.predict([img1_batch, hand1_batch, img2_batch, hand2_batch])[0][0]

        similarities.append(float(similarity))
        true_labels.append(true_label)

    # 다양한 임계값에서 정확도 계산
    thresholds = np.arange(0.1, 1.0, 0.05)
    accuracies = []

    for threshold in thresholds:
        predictions = [1 if s >= threshold else 0 for s in similarities]
        accuracy = sum(1 for p, t in zip(predictions, true_labels) if p == t) / len(predictions)
        accuracies.append(accuracy)

    # 최적 임계값 찾기
    best_idx = np.argmax(accuracies)
    best_threshold = thresholds[best_idx]
    best_accuracy = accuracies[best_idx]

    print(f"최적 임계값: {best_threshold:.2f}, 정확도: {best_accuracy:.4f}")

    return best_threshold


from tensorflow.keras.models import load_model


def resume_training(model_path, csv_path, additional_epochs=40, initial_epoch=10):
    """이전에 학습된 모델에서 학습을 재개합니다 (커스텀 레이어 처리)"""

    # 커스텀 객체 딕셔너리 정의
    custom_objects = {
        'L1DistanceLayer': L1DistanceLayer,
        'contrastive_loss': contrastive_loss
    }

    # 저장된 모델 로드 (커스텀 객체 포함)
    print(f"저장된 모델 로드 중: {model_path}")
    try:
        model = load_model(model_path, custom_objects=custom_objects)
        print("✅ 모델 로드 성공!")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        print("모델을 처음부터 다시 구축합니다...")

        # 모델을 처음부터 재구축
        model, _, _ = build_hybrid_siamese_model()
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        print("✅ 새 모델 구축 완료!")
        initial_epoch = 0  # 처음부터 시작

    # 나머지 코드는 동일...
    # 데이터 분할
    print("데이터 로드 및 분할 중...")
    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df.iloc[:, 2])

    # 임시 CSV 파일 생성
    train_csv = 'train_temp.csv'
    val_csv = 'val_temp.csv'
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    # 데이터 생성기 초기화
    print("데이터 생성기 초기화 중...")
    train_gen = HybridHandwritingDataGenerator(
        train_csv, batch_size=BATCH_SIZE, augment=True, shuffle=True
    )
    val_gen = HybridHandwritingDataGenerator(
        val_csv, batch_size=BATCH_SIZE, augment=False, shuffle=False
    )

    # 콜백 정의
    callbacks = [
        ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        SimpleTrainingMonitor()
    ]

    # 모델 학습 재개
    print(f"\n모델 학습 재개... (에폭: {initial_epoch} -> {initial_epoch + additional_epochs})")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        initial_epoch=initial_epoch,
        epochs=initial_epoch + additional_epochs,
        callbacks=callbacks,
        verbose=1
    )

    # 임시 파일 정리
    os.remove(train_csv)
    os.remove(val_csv)

    print(f"\n추가 학습 완료. 모델 저장 경로: {model_path}")

    return model, history

# 간소화된 훈련 모니터링 콜백
class SimpleTrainingMonitor(tf.keras.callbacks.Callback):
    def __init__(self):
        super(SimpleTrainingMonitor, self).__init__()
        self.start_time = None
        self.epoch_times = []

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        print("훈련 시작...")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        # 에폭 시간 계산
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)

        # 훈련 진행 상황 출력
        val_acc = logs.get('val_accuracy', 0)
        train_acc = logs.get('accuracy', 0)
        val_loss = logs.get('val_loss', 0)
        train_loss = logs.get('loss', 0)

        elapsed = time.time() - self.start_time
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"에폭 {epoch + 1}/{self.params['epochs']} 완료 - {epoch_time:.2f}초")
        print(f"정확도: {train_acc:.4f} / 검증 정확도: {val_acc:.4f}")
        print(f"누적 훈련 시간: {int(hours)}시간 {int(minutes)}분 {int(seconds)}초")

        # 남은 시간 예측
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        remaining_epochs = self.params['epochs'] - (epoch + 1)
        estimated_time = avg_epoch_time * remaining_epochs

        hours, remainder = divmod(estimated_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"예상 남은 시간: {int(hours)}시간 {int(minutes)}분 {int(seconds)}초\n")

    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"\n====== 훈련 완료 ======")
        print(f"총 훈련 시간: {int(hours)}시간 {int(minutes)}분 {int(seconds)}초")
# 메인 함수 - 훈련만 실행
if __name__ == "__main__":
    print("필기체 필적감정 시스템 - 하이브리드 시암 네트워크 훈련")
    print(f"TensorFlow 버전: {tf.__version__}")
    print(f"GPU 사용 가능 여부: {tf.config.list_physical_devices('GPU')}")

    # 모델 훈련

    #model, history = train_hybrid_model_simple(
    #    csv_path=CSV_PATH,
    #    model_path=MODEL_PATH
    #)
    ###
    # 추가 학습 실행
    model, history = resume_training(
        model_path=MODEL_PATH,
        csv_path=CSV_PATH,
        additional_epochs=10,  # 추가할 에폭 수
        initial_epoch=16       # 기존에 학습된 에폭 수
    )



    print("\n훈련 완료!")