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
from scipy.spatial.distance import cdist
import math
import pytesseract
from pytesseract import Output

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
# 📌 STEP 3. 이미지 줄 추출 함수
# -----------------------------------
def extract_lines_from_image(img):
    # 이미지가 컬러인 경우 그레이스케일로 변환
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 이미지 이진화 (적응형 임계값 사용)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 수평 투영 프로필 계산
    h_proj = np.sum(binary, axis=1)

    # 줄 경계 찾기
    line_boundaries = []
    in_line = False
    line_start = 0

    # 최소 줄 높이 (노이즈 필터링 용)
    min_line_height = img.shape[0] * 0.02  # 이미지 높이의 2%

    for i, proj in enumerate(h_proj):
        if not in_line and proj > 0:
            # 줄 시작
            in_line = True
            line_start = i
        elif in_line and (proj == 0 or i == len(h_proj) - 1):
            # 줄 끝
            in_line = False
            line_end = i

            # 최소 높이보다 큰 줄만 저장
            if line_end - line_start > min_line_height:
                line_boundaries.append((line_start, line_end))

    # 원본 이미지에서 줄 추출
    lines = []
    for start, end in line_boundaries:
        # 약간의 여백을 추가하여 줄 추출
        padding = 5
        start_padded = max(0, start - padding)
        end_padded = min(img.shape[0], end + padding)

        line_img = img[start_padded:end_padded, :]
        lines.append(line_img)

    return lines



def extract_lines_with_ocr(img):
    """
    개선된 Tesseract OCR 기반 줄 추출 함수:
    - 빨간 테두리 제거
    - 줄 단위 인식 강화 (psm 6)
    - 줄별 bounding box로 잘라내기
    """
    # 1. 빨간 테두리 제거
    if len(img.shape) == 3 and img.shape[2] == 3:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_red = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255)) + \
                   cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
        img[mask_red > 0] = (255, 255, 255)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 2. Tesseract OCR로 줄 단위 인식
    custom_config = r'--psm 6'
    ocr_data = pytesseract.image_to_data(gray, config=custom_config, output_type=Output.DICT)

    lines = []
    last_line_num = -1
    line_group = []

    for i in range(len(ocr_data['text'])):
        text = ocr_data['text'][i].strip()
        line_num = ocr_data['line_num'][i]

        if text == '':
            continue

        x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]

        if line_num == last_line_num:
            line_group.append((x, y, w, h))
        else:
            if line_group:
                lines.append(line_group)
            line_group = [(x, y, w, h)]
            last_line_num = line_num

    if line_group:
        lines.append(line_group)

    # 3. 각 줄 영역을 잘라서 반환
    line_images = []
    for group in lines:
        xs = [x for x, y, w, h in group]
        ys = [y for x, y, w, h in group]
        ws = [w for x, y, w, h in group]
        hs = [h for x, y, w, h in group]

        x_min = max(0, min(xs) - 5)
        y_min = max(0, min(ys) - 5)
        x_max = min(gray.shape[1], max(x + w for x, w in zip(xs, ws)) + 5)
        y_max = min(gray.shape[0], max(y + h for y, h in zip(ys, hs)) + 5)

        line_img = gray[y_min:y_max, x_min:x_max]
        line_images.append(line_img)

    return line_images


# -----------------------------------
# 📌 STEP 4. 이미지 → 시계열 feature 추출
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
# 📌 STEP 5. Siamese 모델 정의
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
# 📌 STEP 6. 모델 생성 및 가중치 로딩
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
# 📌 STEP 7. 줄 별 비교 함수
# -----------------------------------
def compare_lines(test_lines, ref_lines, threshold=0.2):
    """
    테스트 이미지의 각 줄과 참조 이미지의 각 줄을 비교하여 유사도 행렬 생성
    """
    similarity_matrix = np.zeros((len(test_lines), len(ref_lines)))

    for i, test_line in enumerate(test_lines):
        # 테스트 줄의 특성 추출
        test_feat = np.expand_dims(extract_features_from_image(test_line), axis=0)

        for j, ref_line in enumerate(ref_lines):
            # 참조 줄의 특성 추출
            ref_feat = np.expand_dims(extract_features_from_image(ref_line), axis=0)

            # 거리 계산
            distance = model.predict([test_feat, ref_feat], verbose=0)[0][0]

            # 거리를 유사도로 변환 (거리가 작을수록 유사도가 높음)
            # 거리가 0이면 완전히 일치, 거리가 클수록 차이가 커짐
            # 유사도를 0~1 사이로 정규화 (0: 완전히 다름, 1: 완전히 일치)
            similarity = math.exp(-distance * 5)  # 지수 함수를 사용해 0~1 범위로 매핑
            similarity_matrix[i, j] = similarity

    return similarity_matrix


# -----------------------------------
# 📌 STEP 8. 유사도 행렬 기반 매칭 수행
# -----------------------------------
def find_best_matches(similarity_matrix):
    """
    유사도 행렬에서 각 테스트 줄에 대한 가장 유사한 참조 줄 찾기
    """
    best_matches = []
    # 각 테스트 줄에 대해 최고 유사도와 해당 참조 줄 인덱스 찾기
    for i in range(similarity_matrix.shape[0]):
        best_ref_idx = np.argmax(similarity_matrix[i])
        best_similarity = similarity_matrix[i, best_ref_idx]
        best_matches.append((i, best_ref_idx, best_similarity))

    return best_matches


# (이전 코드 생략)

# -----------------------------------
# 📌 STEP 9. 테스트 이미지와 참조 이미지들 비교 (수정됨)
# -----------------------------------

def extract_pressure_slant_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # 필압 추정: 평균 밝기 (검은색에 가까울수록 필압이 진함)
    pressure_score = np.mean(binary) / 255.0  # 0~1로 정규화

    # 기울기 추정: Hough transform을 사용한 라인 기울기
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    if lines is not None:
        angles = [(theta - np.pi / 2) for rho, theta in lines[:, 0]]
        slant_score = np.mean(np.abs(angles)) / (np.pi / 4)  # 0~1 범위로 정규화
    else:
        slant_score = 0.0

    return pressure_score, slant_score

threshold = 0.5
results = []

for ref_path in reference_img_paths:
    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        print(f"⚠️ 참조 이미지를 로드할 수 없습니다: {ref_path}")
        continue

    test_img = cv2.imread(test_img_path)
    if test_img is None:
        print(f"❌ 테스트 이미지를 로드할 수 없습니다: {test_img_path}")
        exit(1)

    test_lines = extract_lines_with_ocr(test_img)
    ref_lines = extract_lines_with_ocr(ref_img)

    print(f"\n참조 이미지 '{os.path.basename(ref_path)}' 분석 중...")
    print(f"테스트 이미지에서 추출한 줄 수: {len(test_lines)}")
    print(f"참조 이미지에서 추출한 줄 수: {len(ref_lines)}")

    if len(test_lines) == 0 or len(ref_lines) == 0:
        print("⚠️ 줄을 추출할 수 없습니다.")
        continue

    similarity_matrix = compare_lines(test_lines, ref_lines)
    avg_similarity = np.mean(similarity_matrix)
    best_match_avg = np.mean([np.max(similarity_matrix[i]) for i in range(similarity_matrix.shape[0])])

    # 필압/기울기 평균 계산
    pressure_scores = []
    slant_scores = []
    for line in test_lines + ref_lines:
        pressure, slant = extract_pressure_slant_features(line)
        pressure_scores.append(pressure)
        slant_scores.append(slant)
    avg_pressure = np.mean(pressure_scores)
    avg_slant = np.mean(slant_scores)

    is_same = avg_similarity > 0.5
    result = "같은 문서" if is_same else "다른 문서"

    best_matches = find_best_matches(similarity_matrix)

    results.append({
        'reference_image': os.path.basename(ref_path),
        'avg_similarity': avg_similarity,
        'best_match_avg': best_match_avg,
        'avg_pressure': avg_pressure,
        'avg_slant': avg_slant,
        'result': result,
        'is_same': is_same,
        'similarity_matrix': similarity_matrix,
        'ref_img': ref_img,
        'test_lines': test_lines,
        'ref_lines': ref_lines,
        'best_matches': best_matches
    })

results.sort(key=lambda x: x['best_match_avg'], reverse=True)


# (기존 코드 생략)

# -----------------------------------
# 표준 결과 출력의 % 형식 표시로 수정
# -----------------------------------
# 결과가 없는 경우 처리
if not results:
    print("❌ 비교할 결과가 없습니다.")
    exit(1)

# 결과 변수 설정 및 % 계산
best_result = results[0]
similarity_percent = best_result['avg_similarity'] * 100
pressure_percent = best_result['avg_pressure'] * 100
slant_percent = best_result['avg_slant'] * 100
best_match_percent = best_result['best_match_avg'] * 100

# 결과 출력
print(f"\n🏆 최고 유사도 결과: {best_result['reference_image']}")
print(f"줄 매칭 평균 유사도: {best_match_percent:.2f}%")
print(f"전체 유사도 평균: {similarity_percent:.2f}%")
print(f"평균 필압(Pressure): {pressure_percent:.2f}%")
print(f"평균 기울기(Slant): {slant_percent:.2f}%")
print(f"판정 결과: {best_result['result']}")

# 히트맵 + 테스트/참조 이미지 시각화
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.imshow(best_result['similarity_matrix'], cmap='viridis', aspect='auto')
plt.colorbar(label='Similarity')
plt.title(f"Line Similarity Matrix: {best_result['reference_image']}")
plt.xlabel('Reference Lines')
plt.ylabel('Test Lines')

plt.subplot(2, 2, 2)
if len(test_img.shape) == 3:
    plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
else:
    plt.imshow(test_img, cmap='gray')
plt.title("Test Image")
plt.axis('off')

plt.subplot(2, 2, 3)
if len(best_result['ref_img'].shape) == 3:
    plt.imshow(cv2.cvtColor(best_result['ref_img'], cv2.COLOR_BGR2RGB))
else:
    plt.imshow(best_result['ref_img'], cmap='gray')
plt.title(f"Reference Image: {best_result['reference_image']}")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.text(0.5, 0.5,
         f"Best Match Average: {best_match_percent:.2f}%\nSimilarity: {similarity_percent:.2f}%\nPressure: {pressure_percent:.2f}%\nSlant: {slant_percent:.2f}%\nResult: {best_result['result']}",
         horizontalalignment='center', verticalalignment='center', fontsize=12)
plt.axis('off')
plt.gca().set_facecolor((0.9, 1, 0.9) if best_result['is_same'] else (1, 0.9, 0.9))
plt.tight_layout()

# 줄 매칭 시각화
num_matches = min(5, len(best_result['best_matches']))
plt.figure(figsize=(15, 3 * num_matches))
for i in range(num_matches):
    match = best_result['best_matches'][i]
    test_idx, ref_idx, similarity = match

    plt.subplot(num_matches, 2, i * 2 + 1)
    plt.imshow(best_result['test_lines'][test_idx], cmap='gray')
    plt.title(f"Test Line {test_idx + 1}")
    plt.axis('off')

    plt.subplot(num_matches, 2, i * 2 + 2)
    plt.imshow(best_result['ref_lines'][ref_idx], cmap='gray')
    plt.title(f"Matched Ref Line {ref_idx + 1} (Similarity: {similarity * 100:.2f}%)")
    plt.axis('off')
plt.tight_layout()

# 유사도/필압/기울기 % 그래프
plt.figure(figsize=(6, 4))
metrics = ['Similarity', 'Pressure', 'Slant']
values = [similarity_percent, pressure_percent, slant_percent]
bars = plt.bar(metrics, values)
plt.ylim(0, 100)
plt.title('평균 유사도 / 필압 / 기울기 (%)')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1,
             f"{height:.1f}%", ha='center', va='bottom')
plt.tight_layout()
plt.show()

# 요약 출력
same_doc_count = sum(1 for r in results if r['is_same'])
print(f"\n결과 요약: 총 {len(results)}개 참조 이미지 중 {same_doc_count}개가 테스트 이미지와 같은 문서로 판별됨")
