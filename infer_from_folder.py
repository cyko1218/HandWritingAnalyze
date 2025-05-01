# -----------------------------------
# 📌 STEP 0. 임포트
# -----------------------------------
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, Conv1D, BatchNormalization, LSTM, Dense, Lambda, Reshape
from tensorflow.keras import backend as K

# -----------------------------------
# 📌 STEP 1. 한글 폰트 설정
# -----------------------------------
# 한글 폰트 설정 (맥OS)
try:
    font_path = '/System/Library/Fonts/AppleSDGothicNeo.ttc'  # 맥OS 기본 한글 폰트
    if os.path.exists(font_path):
        plt.rcParams['font.family'] = 'AppleGothic'
    else:
        # 다른 한글 폰트 검색
        korean_fonts = [f for f in fm.findSystemFonts(fontpaths=None) if any(name in f for name in
                                                                             ['Gothic', 'Batang', 'Myeongjo', 'Gulim',
                                                                              'Dotum', 'Malgun', 'NanumGothic',
                                                                              'NanumMyeongjo'])]
        if korean_fonts:
            plt.rcParams['font.family'] = fm.FontProperties(fname=korean_fonts[0]).get_name()
        else:
            print("⚠️ 한글 폰트를 찾을 수 없습니다. 한글이 깨져 보일 수 있습니다.")
except Exception as e:
    print(f"폰트 설정 중 오류 발생: {e}")
    print("⚠️ 폰트 문제로 한글이 깨져 보일 수 있습니다.")

# -----------------------------------
# 📌 STEP 2. 경로 설정 및 검증
# -----------------------------------
# 참조 이미지가 있는 폴더와 테스트 이미지가 있는 폴더 경로
reference_dir = 'reference_samples'
test_dir = 'test_samples'
model_path = 'our.net.hdf5'  # 현재 디렉토리에 모델 파일이 있는 경우

# 디렉토리 존재 여부 확인
if not os.path.exists(reference_dir):
    print(f"⚠️ 참조 디렉토리가 존재하지 않습니다: {reference_dir}")
    os.makedirs(reference_dir)
    print(f"디렉토리를 생성했습니다. 참조 이미지를 추가해주세요.")
    exit(1)

if not os.path.exists(test_dir):
    print(f"⚠️ 테스트 디렉토리가 존재하지 않습니다: {test_dir}")
    os.makedirs(test_dir)
    print(f"디렉토리를 생성했습니다. 테스트 이미지를 추가해주세요.")
    exit(1)

# 참조 이미지 경로 목록
reference_img_paths = [os.path.join(reference_dir, f) for f in os.listdir(reference_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not reference_img_paths:
    print(f"⚠️ 참조 디렉토리에 이미지 파일이 없습니다: {reference_dir}")
    print("지원 형식: PNG, JPG, JPEG")
    exit(1)

# 테스트 이미지 경로 목록
test_img_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not test_img_paths:
    print(f"⚠️ 테스트 디렉토리에 이미지 파일이 없습니다: {test_dir}")
    print("지원 형식: PNG, JPG, JPEG")
    exit(1)

# 테스트 이미지 선택 (첫 번째 이미지 사용)
test_img_path = test_img_paths[0]
print(f"테스트 이미지: {os.path.basename(test_img_path)}")
print(f"참조 이미지 수: {len(reference_img_paths)}")

# 모델 파일 존재 확인
if not os.path.exists(model_path):
    print(f"⚠️ 모델 파일이 지정된 경로에 존재하지 않습니다: {model_path}")
    # 프로젝트 폴더에서 hdf5 파일 찾기
    found_models = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.hdf5') or file.endswith('.h5'):
                found_models.append(os.path.join(root, file))

    if found_models:
        print("💡 다음 모델 파일을 찾았습니다:")
        for i, found_model in enumerate(found_models):
            print(f"  {i + 1}. {found_model}")

        choice = input("사용할 모델 번호를 입력하세요 (또는 q로 종료): ")
        if choice.lower() == 'q':
            exit(1)
        try:
            model_path = found_models[int(choice) - 1]
            print(f"선택한 모델: {model_path}")
        except (ValueError, IndexError):
            print("잘못된 선택입니다. 프로그램을 종료합니다.")
            exit(1)
    else:
        print("💡 프로젝트 폴더에서 .hdf5 또는 .h5 파일을 찾을 수 없습니다.")
        custom_path = input("모델 파일의 전체 경로를 입력하세요 (또는 q로 종료): ")
        if custom_path.lower() == 'q':
            exit(1)
        model_path = custom_path
        if not os.path.exists(model_path):
            print(f"입력한 경로에 파일이 존재하지 않습니다: {model_path}")
            print("프로그램을 종료합니다.")
            exit(1)


# -----------------------------------
# 📌 STEP 3. 이미지 → 시계열 feature 추출
# -----------------------------------
def extract_features_from_image(img):
    # 이미지가 컬러인 경우 그레이스케일로 변환
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (300, 300))
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 컨투어가 없는 경우 처리
    if not contours:
        print("⚠️ 컨투어를 찾을 수 없습니다. 빈 이미지인지 확인하세요.")
        # 빈 컨투어 대신 임의의 노이즈 포인트 생성
        contour_points = np.random.randint(0, 300, size=(150, 2))
    else:
        contour_points = np.concatenate(contours, axis=0).squeeze()

        # contour_points가 1차원인 경우 (단일 포인트)
        if contour_points.ndim == 1:
            contour_points = contour_points.reshape(1, 2)

        # 포인트가 부족한 경우 패딩
        if len(contour_points) < 150:
            # 마지막 포인트를 복제하여 패딩
            pad_length = 150 - len(contour_points)
            contour_points = np.vstack([contour_points,
                                        np.tile(contour_points[-1], (pad_length, 1))])

    # 포인트가 충분한 경우 랜덤 샘플링
    if len(contour_points) > 150:
        selected = contour_points[np.random.choice(len(contour_points), 150, replace=False)]
    else:
        selected = contour_points[:150]

    # 특성 추출
    features = np.stack([
        selected[:, 0], selected[:, 1],  # x, y
        np.gradient(selected[:, 0]),  # dx
        np.gradient(selected[:, 1]),  # dy
    ], axis=-1)  # (150, 4)

    # 24개 특성으로 확장
    while features.shape[1] < 24:
        features = np.concatenate([features, features], axis=1)
    features = features[:, :24]

    # 정규화
    std = np.std(features, axis=0)
    mean = np.mean(features, axis=0)
    features = (features - mean) / (std + 1e-8)

    return features.astype(np.float32).reshape(150, 24, 1)


# -----------------------------------
# 📌 STEP 4. Siamese 모델 정의
# -----------------------------------
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def net(input_shape, timeseries_n, feature_l):
    input_layer = Input(shape=input_shape)  # (150, 24, 1)

    conv1 = Conv1D(32, 5, activation='gelu', padding='same')
    x = TimeDistributed(conv1)(input_layer)  # (150, 24, 32)
    x = BatchNormalization()(x)

    conv2 = Conv1D(1, 1, activation='gelu')
    x = TimeDistributed(conv2)(x)  # (150, 24, 1)
    x = BatchNormalization()(x)

    x = Reshape((timeseries_n, feature_l))(x)  # (150, 24)

    x = LSTM(feature_l, return_sequences=True)(x)
    x = LSTM(32, return_sequences=False)(x)

    x = Dense(8, activation='gelu')(x)

    return Model(inputs=input_layer, outputs=x)


def create_full_model(input_shape):
    base_network = net(input_shape, timeseries_n=150, feature_l=24)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    vec_a = base_network(input_a)
    vec_b = base_network(input_b)

    distance = Lambda(euclidean_distance)([vec_a, vec_b])

    return Model(inputs=[input_a, input_b], outputs=distance)


# -----------------------------------
# 📌 STEP 5. 모델 생성 및 가중치 로딩
# -----------------------------------
input_shape = (150, 24, 1)
model = create_full_model(input_shape)

# 모델 가중치 로드
try:
    model.load_weights(model_path)
    print(f"✅ 모델 가중치를 성공적으로 로드했습니다: {model_path}")
except Exception as e:
    print(f"❌ 모델 가중치 로드 실패: {e}")
    exit(1)

# -----------------------------------
# 📌 STEP 6. 테스트 이미지와 참조 이미지들 비교
# -----------------------------------
# 테스트 이미지 로드 및 특성 추출
test_img = cv2.imread(test_img_path)
if test_img is None:
    print(f"❌ 테스트 이미지를 로드할 수 없습니다: {test_img_path}")
    exit(1)

test_feat = np.expand_dims(extract_features_from_image(test_img), axis=0)  # (1, 150, 24, 1)

# 임계값 설정
threshold = 0.2

# 결과 저장을 위한 리스트
results = []

# 각 참조 이미지와 비교
for ref_path in reference_img_paths:
    # 참조 이미지 로드
    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        print(f"⚠️ 참조 이미지를 로드할 수 없습니다: {ref_path}")
        continue

    # 특성 추출
    ref_feat = np.expand_dims(extract_features_from_image(ref_img), axis=0)

    # 거리 계산
    distance = model.predict([test_feat, ref_feat], verbose=0)[0][0]

    # 같은 사람인지 판별
    is_same = distance < threshold
    result = "같은 사람" if is_same else "다른 사람"

    # 결과 저장
    results.append({
        'reference_image': os.path.basename(ref_path),
        'distance': distance,
        'result': result,
        'is_same': is_same,
        'ref_img': ref_img if len(ref_img.shape) == 2 else cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    })

    print(f"참조 이미지 '{os.path.basename(ref_path)}' 비교 결과: 거리={distance:.4f}, {result}")

# 결과를 거리순으로 정렬
results.sort(key=lambda x: x['distance'])

# -----------------------------------
# 📌 STEP 7. 결과 시각화
# -----------------------------------
# 결과가 없는 경우 처리
if not results:
    print("❌ 비교할 결과가 없습니다.")
    exit(1)

# 테스트 이미지를 그레이스케일로 변환
if len(test_img.shape) == 3:
    test_img_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
else:
    test_img_gray = test_img

# 상위 5개 또는 전체 결과 (더 적은 쪽) 표시
display_count = min(5, len(results))

plt.figure(figsize=(15, 3 * display_count))

# 테스트 이미지 (항상 왼쪽에 표시)
plt.subplot(display_count, 3, 1)
plt.imshow(test_img_gray, cmap='gray')
plt.title("Test Image", fontsize=12)  # 영어로 표시하여 폰트 문제 회피
plt.axis('off')

# 각 참조 이미지 및 결과 표시
for i in range(display_count):
    result = results[i]

    # 참조 이미지
    plt.subplot(display_count, 3, i * 3 + 2)
    plt.imshow(result['ref_img'], cmap='gray')
    plt.title(f"Reference: {result['reference_image']}", fontsize=10)  # 영어로 표시
    plt.axis('off')

    # 결과 텍스트
    plt.subplot(display_count, 3, i * 3 + 3)
    result_text = f"Distance: {result['distance']:.4f}\nResult: "
    result_text += "Same Person" if result['is_same'] else "Different Person"  # 영어로 표시

    plt.text(0.5, 0.5, result_text,
             horizontalalignment='center', verticalalignment='center', fontsize=12)
    plt.axis('off')

    # 배경색 설정 (같은 사람이면 연한 녹색, 다른 사람이면 연한 빨간색)
    if result['is_same']:
        plt.gca().set_facecolor((0.9, 1, 0.9))  # 연한 녹색
    else:
        plt.gca().set_facecolor((1, 0.9, 0.9))  # 연한 빨간색

plt.tight_layout()
plt.show()

# 종합 결과 출력
same_person_count = sum(1 for r in results if r['is_same'])
print(f"\n결과 요약: 총 {len(results)}개 참조 이미지 중 {same_person_count}개가 테스트 이미지와 같은 사람으로 판별됨")

# 가장 유사한 참조 이미지 결과
if results:
    best_match = results[0]
    print(
        f"가장 유사한 참조 이미지: {best_match['reference_image']} (거리: {best_match['distance']:.4f}, 결과: {best_match['result']})")