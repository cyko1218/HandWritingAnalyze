import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.feature import graycomatrix, graycoprops
from PIL import Image
import pandas as pd


def set_korean_font():
    """한글 폰트 설정 함수"""
    try:
        # Windows 환경
        if os.name == 'nt':
            font_path = 'c:/Windows/Fonts/malgun.ttf'  # 맑은 고딕 폰트 사용
            if not os.path.exists(font_path):
                # 다른 한글 폰트 시도
                font_path = 'c:/Windows/Fonts/gulim.ttc'  # 굴림체 시도
        # macOS 환경
        elif os.name == 'posix' and os.uname().sysname == 'Darwin':
            font_path = '/System/Library/Fonts/AppleSDGothicNeo.ttc'  # Apple SD Gothic Neo 폰트
            if not os.path.exists(font_path):
                font_path = '/Library/Fonts/AppleGothic.ttf'  # Apple Gothic 폰트 시도
        # Linux 환경
        else:
            # Linux 기본 한글 폰트 경로들
            possible_paths = [
                '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
                '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf',
                '/usr/share/fonts/nanum/NanumGothic.ttf'
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    font_path = path
                    break
            else:
                print("한글 폰트를 찾을 수 없습니다. 기본 유니코드 폰트를 사용합니다.")
                return False

        # 폰트 설정
        from matplotlib import font_manager, rc
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        plt.rc('font', family=font_name)
        # 음수 표시가 깨지는 문제 해결
        plt.rc('axes', unicode_minus=False)

        print(f"한글 폰트 '{font_name}'이(가) 성공적으로 설정되었습니다.")
        return True

    except Exception as e:
        print(f"한글 폰트 설정 중 오류 발생: {str(e)}")
        print("한글이 제대로 표시되지 않을 수 있습니다.")
        return False


class HandwritingAnalyzer:
    def __init__(self):
        """필체 분석기 초기화"""
        # 4가지 핵심 특성에 대한 임계값 설정
        self.thresholds = {
            'size': 150,  # 글자 높이 기준 (픽셀)
            'pressure': 200,  # 필압 강도 기준 (0-255)
            'slant': 9,  # 기울기 각도 기준 (도)
            'roundness': 0.2  # 둥근 정도 기준 (0-1)
        }

        # 성격 특성 매핑
        self.personality_traits = {
            'size': {
                'small': "내향적, 치밀함, 절약 정신, 조심스러움",
                'large': "외향적, 표현력 강함, 자신감 있음, 사회성 있음"
            },
            'pressure': {
                'strong': "의지가 굳음, 자기주장이 강함, 정신력이 강함, 활력이 있음",
                'weak': "유순함, 수줍음, 에너지가 약함, 민감함"
            },
            'slant': {
                'rightward': "낙관적, 열정적, 희망적, 진취적",
                'leftward': "비관적, 감정표현을 잘 안함, 차가움, 비판적"
            },
            'shape': {
                'angular': "원칙을 중시, 정직함, 고집스러움, 규범을 잘 지킴",
                'round': "사고가 유연함, 상상력이 풍부, 원만함, 합리적임"
            }
        }

        # 16가지 성격 유형 정의
        self.personality_types = {
            ('small', 'strong', 'rightward', 'angular'): "원칙주의적 완벽주의자 - 내향적이지만 강한 의지와 낙관적인 성향, 원칙을 중시함",
            ('small', 'strong', 'rightward', 'round'): "창의적 전문가 - 내향적이지만 의지가 강하고 낙관적이며 유연한 사고를 갖춤",
            ('small', 'strong', 'leftward', 'angular'): "비판적 분석가 - 내향적이고 의지가 강하며 비판적이고 원칙적인 성향",
            ('small', 'strong', 'leftward', 'round'): "신중한 혁신가 - 내향적이고 의지가 강하며 비판적이지만 유연한 사고",
            ('small', 'weak', 'rightward', 'angular'): "충실한 조력자 - 내향적이고 유순하지만 낙관적이며 원칙적인 성향",
            ('small', 'weak', 'rightward', 'round'): "조용한 창의가 - 내향적이고 유순하며 낙관적이고 유연한 사고",
            ('small', 'weak', 'leftward', 'angular'): "신중한 비평가 - 내향적이고 유순하며 비판적이고 원칙적인 성향",
            ('small', 'weak', 'leftward', 'round'): "섬세한 관찰자 - 내향적이고 유순하며 비판적이지만 유연한 사고",
            ('large', 'strong', 'rightward', 'angular'): "리더형 실행가 - 외향적이고 자기주장이 강하며 낙관적이고 원칙적인 성향",
            ('large', 'strong', 'rightward', 'round'): "열정적 비전리더 - 외향적이고 자기주장이 강하며 낙관적이고 유연한 사고",
            ('large', 'strong', 'leftward', 'angular'): "도전적 실용주의자 - 외향적이고 자기주장이 강하며 비판적이고 원칙적인 성향",
            ('large', 'strong', 'leftward', 'round'): "전략적 개혁가 - 외향적이고 자기주장이 강하며 비판적이지만 유연한 사고",
            ('large', 'weak', 'rightward', 'angular'): "사교적 조정자 - 외향적이지만 유순하며 낙관적이고 원칙적인 성향",
            ('large', 'weak', 'rightward', 'round'): "자유로운 영감가 - 외향적이지만 유순하며 낙관적이고 유연한 사고",
            ('large', 'weak', 'leftward', 'angular'): "사실적 대변인 - 외향적이지만 유순하며 비판적이고 원칙적인 성향",
            ('large', 'weak', 'leftward', 'round'): "적응적 표현가 - 외향적이지만 유순하며 비판적이지만 유연한 사고"
        }

    def preprocess_image(self, image_path):
        """이미지 전처리 함수"""
        try:
            # 이미지 로드
            if isinstance(image_path, str):
                # 파일 경로인 경우
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
            else:
                # 이미 배열 형태인 경우 (예: 업로드된 이미지)
                img = image_path

            # 그레이스케일 변환
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 노이즈 제거
            denoised = cv2.GaussianBlur(gray, (5, 5), 0)

            # 이진화 (Otsu 알고리즘 사용)
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # 모폴로지 연산으로 노이즈 제거
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # 저장할 이미지들
            processed_images = {
                'original': img,
                'gray': gray,
                'binary': binary
            }

            return processed_images

        except Exception as e:
            print(f"이미지 전처리 중 오류 발생: {str(e)}")
            return None

    def extract_features(self, processed_images):
        """4가지 핵심 필적 특성 추출 함수"""
        try:
            binary = processed_images['binary']
            gray = processed_images['gray']
            features = {}

            # 1. 글씨 크기 분석
            contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            heights = []
            for cnt in contours:
                if cv2.contourArea(cnt) > 30:  # 작은 노이즈 필터링
                    x, y, w, h = cv2.boundingRect(cnt)
                    heights.append(h)

            features['avg_height'] = np.mean(heights) if heights else 0

            # 2. 필압 분석
            # 이진화된 이미지에서 글자 부분의 원본 그레이스케일 값 측정
            if np.max(binary) > 0:
                mask = binary > 0
                if np.sum(mask) > 0:
                    # 글씨 부분의 평균 강도 계산 (낮을수록 필압이 강함)
                    # 반전시켜서 높은 값이 강한 필압을 나타내도록 함
                    pressure_values = 255 - gray[mask]
                    features['pressure'] = np.mean(pressure_values)
                else:
                    features['pressure'] = 0
            else:
                features['pressure'] = 0

            # 3. 기울기 분석 - 글자 전체 기울기 측정 (PCA 방식 사용)
            def analyze_slant(binary_image):
                # 윤곽선 검출
                contours, _ = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 모든 글자의 기울기 각도를 저장할 리스트
                all_angles = []

                # 각 글자(윤곽선)별로 기울기 분석
                for cnt in contours:
                    # 작은 노이즈는 건너뛰기
                    if cv2.contourArea(cnt) < 100:
                        continue

                    # 글자의 경계 상자 정보
                    rect = cv2.minAreaRect(cnt)
                    width, height = rect[1]
                    angle = rect[2]

                    # 경계 상자가 세로로 긴 경우(높이>너비) 각도 보정
                    if height > width:
                        angle = angle - 90

                    # 각도 범위를 -45°~45°로 조정
                    while angle < -45:
                        angle += 90
                    while angle > 45:
                        angle -= 90

                    all_angles.append(angle)

                # 글자가 없는 경우
                if not all_angles:
                    return 0

                # 각도의 중앙값 사용 (이상치에 덜 민감)
                return np.median(all_angles)

            def analyze_slant_by_top_points(binary_image):
                """글자의 상단점을 연결하여 기울기를 분석하는 함수"""
                # 윤곽선 찾기
                contours, _ = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 각 글자 윤곽선의 경계 상자와 상단점 찾기
                letter_tops = []

                for cnt in contours:
                    # 작은 노이즈 필터링
                    if cv2.contourArea(cnt) < 50:
                        continue

                    # 경계 상자 구하기
                    x, y, w, h = cv2.boundingRect(cnt)

                    # 글자의 상단 중심점 (x + w/2, y)
                    top_center_x = x + w / 2
                    top_y = y

                    # 글자가 충분히 큰 경우에만 포함
                    if h > 10:
                        letter_tops.append((top_center_x, top_y))

                # 글자가 충분하지 않으면 0 반환
                if len(letter_tops) < 2:
                    return 0

                # x좌표로 정렬 (왼쪽에서 오른쪽으로)
                letter_tops.sort(key=lambda p: p[0])

                # 디버깅: 상단점 정보 출력
                # print(f"정렬된 글자 상단점: {letter_tops}")

                # 선형 회귀로 기울기 계산
                x_coords = np.array([p[0] for p in letter_tops])
                y_coords = np.array([p[1] for p in letter_tops])

                # 선형 회귀로 기울기 계산 (y = mx + b)
                slope, intercept = np.polyfit(x_coords, y_coords, 1)

                # 이미지 좌표계에서는 y가 아래로 증가하므로,
                # 양의 기울기는 우하향, 음의 기울기는 우상향을 의미
                # 따라서 부호를 반전시켜 직관적인 해석이 가능하게 함
                angle = np.arctan(-slope) * 180 / np.pi

                # 시각화를 위한 선 좌표 계산 (옵션)
                x_start = min(x_coords)
                x_end = max(x_coords)
                y_start = slope * x_start + intercept
                y_end = slope * x_end + intercept
                line_coords = ((int(x_start), int(y_start)), (int(x_end), int(y_end)))

                return angle, line_coords, letter_tops

            # 필적 특성 추출 함수 내에서 사용 (extract_features 함수의 일부)
            try:
                # 상단점 기반 기울기 분석
                slant_result = analyze_slant_by_top_points(binary)

                if isinstance(slant_result, tuple):
                    # 추가 정보가 반환된 경우
                    slant_angle, line_coords, letter_tops = slant_result

                    # 시각화를 위해 저장 (선택 사항)
                    features['slant_line'] = line_coords
                    features['letter_tops'] = letter_tops
                else:
                    # 각도만 반환된 경우
                    slant_angle = slant_result

                # 결과 저장
                features['slant_angle'] = slant_angle

                # 디버깅 출력
                #print(f"상단점 연결 기울기 각도: {slant_angle:.2f}°")
                #print(f"기울기 방향: {'우상향' if slant_angle > 0 else '우하향' if slant_angle < 0 else '수직'}")

            except Exception as e:
                print(f"기울기 분석 중 오류 발생: {str(e)}")
                features['slant_angle'] = 0
            # 4. 글자 모양 분석 (각진 정도 vs 둥근 정도)
            roundness_values = []
            for cnt in contours:
                if cv2.contourArea(cnt) > 30:
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter > 0:
                        area = cv2.contourArea(cnt)
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        roundness_values.append(circularity)

            features['roundness'] = np.mean(roundness_values) if roundness_values else 0

            return features

        except Exception as e:
            print(f"특징 추출 중 오류 발생: {str(e)}")
            return None

    def analyze_personality(self, features):
        """추출된 특성을 바탕으로 성격 분석"""
        try:
            categories = {}
            measured_values = {}  # 측정값 저장용 딕셔너리

            # 1. 글씨 크기에 따른 분류
            measured_values['size'] = features['avg_height']
            categories['size'] = 'small' if features['avg_height'] < self.thresholds['size'] else 'large'
            #print(f"글씨 크기 측정값: {features['avg_height']:.2f}px (임계값: {self.thresholds['size']}px)")

            # 2. 필압에 따른 분류
            measured_values['pressure'] = features['pressure']
            categories['pressure'] = 'strong' if features['pressure'] > self.thresholds['pressure'] else 'weak'
            #print(f"필압 측정값: {features['pressure']:.2f} (임계값: {self.thresholds['pressure']})")

            # 3. 기울기에 따른 분류
            measured_values['slant'] = features['slant_angle']
            categories['slant'] = 'rightward' if features['slant_angle'] > self.thresholds['slant'] else 'leftward'
            #print(f"기울기 측정값: {features['slant_angle']:.2f}° (임계값: {self.thresholds['slant']}°)")

            # 4. 글자 모양에 따른 분류
            measured_values['roundness'] = features['roundness']
            categories['shape'] = 'round' if features['roundness'] > self.thresholds['roundness'] else 'angular'
            #print(f"둥근 정도 측정값: {features['roundness']:.2f} (임계값: {self.thresholds['roundness']})")

            # 성격 특성 추출
            personality = {
                'categories': categories,
                'measured_values': measured_values,  # 측정값 저장
                'traits': {
                    'size': self.personality_traits['size'][categories['size']],
                    'pressure': self.personality_traits['pressure'][categories['pressure']],
                    'slant': self.personality_traits['slant'][categories['slant']],
                    'shape': self.personality_traits['shape'][categories['shape']]
                }
            }

            # 16가지 유형 중 해당하는 유형 찾기
            type_key = (categories['size'], categories['pressure'], categories['slant'], categories['shape'])
            personality['type'] = self.personality_types.get(type_key, "알 수 없는 유형")

            # 종합 결과 출력
            print("\n=== 필체 분석 결과 ===")
            print(f"글씨 크기: {'작음' if categories['size'] == 'small' else '큼'} ({measured_values['size']:.2f}px)")
            print(f"필압: {'강함' if categories['pressure'] == 'strong' else '약함'} ({measured_values['pressure']:.2f})")
            print(f"기울기: {'우상향' if categories['slant'] == 'rightward' else '우하향'} ({measured_values['slant']:.2f}°)")
            print(f"글자 모양: {'둥글다' if categories['shape'] == 'round' else '각지다'} ({measured_values['roundness']:.2f})")
            print(f"\n성격 유형: {personality['type']}")
            # 요약 문장 생성
            personality['summary'] = self.generate_summary_paragraph(personality['traits'])

            return personality

        except Exception as e:
            print(f"성격 분석 중 오류 발생: {str(e)}")
            return None

    def visualize_analysis(self, processed_images, features, personality):
        """분석 결과 시각화"""
        # 한글 폰트 설정
        set_korean_font()

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 6)

        # 원본 이미지
        ax1 = fig.add_subplot(gs[0, :3])
        ax1.imshow(cv2.cvtColor(processed_images['original'], cv2.COLOR_BGR2RGB))
        ax1.set_title('원본 이미지')
        ax1.axis('off')

        # 이진화 이미지
        ax2 = fig.add_subplot(gs[0, 3:])
        ax2.imshow(processed_images['binary'], cmap='gray')
        ax2.set_title('이진화 이미지')
        ax2.axis('off')

        # 특성 바 차트
        ax3 = fig.add_subplot(gs[1, :3])

        feature_names = ['글씨 크기', '필압', '기울기', '둥근 정도']
        feature_values = [
            features['avg_height'] / (self.thresholds['size'] * 2),  # 0-1 정규화
            features['pressure'] / 255,  # 0-1 정규화
            (features['slant_angle'] + 45) / 90,  # -45~45 -> 0-1 정규화
            features['roundness']  # 이미 0-1 범위
        ]

        # 값 클리핑 (0-1 범위로 제한)
        feature_values = [max(0, min(1, val)) for val in feature_values]

        # 바 차트 생성
        bars = ax3.barh(feature_names, feature_values, color='skyblue')
        ax3.set_xlim(0, 1)
        ax3.set_title('필적 특성 측정')

        # 임계값 표시
        threshold_values = [
            self.thresholds['size'] / (self.thresholds['size'] * 2),
            self.thresholds['pressure'] / 255,
            (self.thresholds['slant'] + 45) / 90,
            self.thresholds['roundness']
        ]

        for i, threshold in enumerate(threshold_values):
            ax3.axvline(x=threshold, ymin=i / len(feature_names), ymax=(i + 1) / len(feature_names),
                        color='red', linestyle='--', alpha=0.7)

        # 결과 텍스트
        categories = personality['categories']
        ax4 = fig.add_subplot(gs[1, 3:])
        ax4.axis('off')

        # 카테고리 텍스트 생성
        category_text = "\n".join([
            f"글씨 크기: {'작음' if categories['size'] == 'small' else '큼'}",
            f"필압: {'강함' if categories['pressure'] == 'strong' else '약함'}",
            f"기울기: {'우상향' if categories['slant'] == 'rightward' else '우하향'}",
            f"글자 모양: {'둥글다' if categories['shape'] == 'round' else '각지다'}"
        ])

        ax4.text(0, 0.7, category_text, fontsize=14)
        ax4.set_title('필적 특성 분류 결과')

        # 성격 유형 및 특성
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')

        # 성격 유형 정보
        personality_info = f"[ 성격 유형 ]\n{personality['type']}\n\n"

        # 각 특성별 성격 특성
        traits_info = "[ 세부 성격 특성 ]\n"
        for feature, trait in personality['traits'].items():
            if feature == 'size':
                feature_name = "글씨 크기"
            elif feature == 'pressure':
                feature_name = "필압"
            elif feature == 'slant':
                feature_name = "기울기"
            elif feature == 'shape':
                feature_name = "글자 모양"
            else:
                feature_name = feature

            traits_info += f"• {feature_name}: {trait}\n"

        # 텍스트 출력
        ax5.text(0, 0.7, personality_info + traits_info, fontsize=14, linespacing=1.5)
        ax5.set_title('성격 분석 결과')

        plt.tight_layout()
        return fig

    def analyze_image(self, image_path, visualize=True):
        """이미지 분석 전체 파이프라인"""
        # 1. 이미지 전처리
        processed_images = self.preprocess_image(image_path)
        if processed_images is None:
            return {"error": "이미지 전처리 실패"}

        # 2. 특성 추출
        features = self.extract_features(processed_images)
        if features is None:
            return {"error": "특성 추출 실패"}

        # 3. 성격 분석
        personality = self.analyze_personality(features)
        if personality is None:
            return {"error": "성격 분석 실패"}

        # 4. 결과 시각화 (선택적)
        if visualize:
            visualization = self.visualize_analysis(processed_images, features, personality)
            return {
                "features": features,
                "personality": personality,
                "visualization": visualization
            }

        return {
            "features": features,
            "personality": personality
        }

    def generate_summary_paragraph(self, traits):
        parts = []

        # 글씨 크기
        if '내향적' in traits['size']:
            parts.append("내향적이고 치밀하며 조심스러운 성향을 지녔고")
        else:
            parts.append("외향적이고 자신감 있으며 표현력이 뛰어난 성향을 보이고")

        # 필압
        if '의지가 굳음' in traits['pressure']:
            parts.append("강한 의지와 자기주장을 바탕으로 활력 있게 자신의 생각을 표현합니다.")
        else:
            parts.append("유순하고 민감하며 에너지는 다소 낮지만 주변에 조화롭게 어울립니다.")

        # 기울기
        if '낙관적' in traits['slant']:
            parts.append("낙관적이고 진취적인 태도를 가지고 있으며")
        else:
            parts.append("감정을 드러내는 데 소극적이고 신중하며 다소 비판적인 관점을 지닙니다.")

        # 글자 모양
        if '사고가 유연함' in traits['shape']:
            parts.append("사고가 유연하고 상상력이 풍부하여 원만하고 합리적인 관계를 지향하는 경향이 있습니다.")
        else:
            parts.append("정직하고 규범을 중시하며 단호하고 원칙적인 태도를 유지합니다.")

        return " ".join(parts)


# 앱 사용 예시
def main(image_path):
    """분석 실행 함수"""
    # 한글 폰트 설정
    set_korean_font()

    # 분석기 초기화
    analyzer = HandwritingAnalyzer()

    # 이미지가 존재하는지 확인
    if not os.path.exists(image_path):
        print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        return

    # 이미지 분석 실행
    result = analyzer.analyze_image(image_path)

    if "error" in result:
        print(f"분석 중 오류 발생: {result['error']}")
        return

    # 결과 출력
    print("\n=== 필적 성격 분석 결과 ===")
    print("\n성격 유형:")
    print(result["personality"]["type"])

    print("\n특성 카테고리:")
    category_display = {
        'size': {'small': '작음', 'large': '큼'},
        'pressure': {'strong': '강함', 'weak': '약함'},
        'slant': {'rightward': '우상향', 'leftward': '우하향'},
        'shape': {'round': '둥글다', 'angular': '각지다'}
    }
    for feature, category in result["personality"]["categories"].items():
        display = category_display.get(feature, {}).get(category, category)
        print(f"{feature}: {display}")

    print("\n성격 특성:")
    category_display = {
        'size': {'small': '작음', 'large': '큼'},
        'pressure': {'strong': '강함', 'weak': '약함'},
        'slant': {'rightward': '우상향', 'leftward': '우하향'},
        'shape': {'round': '둥글다', 'angular': '각지다'}
    }

    for feature in ['size', 'pressure', 'slant', 'shape']:
        category = result["personality"]["categories"][feature]
        label = category_display[feature][category]
        trait = result["personality"]["traits"][feature]
        print(f"{label}: {trait}")

    print("\n성격 요약:")
    print(result["personality"]["summary"])

    # 시각화 결과 표시
    if "visualization" in result:
        plt.show()

    return result


if __name__ == "__main__":
    # 테스트 이미지 경로 (실제 경로로 변경 필요)
    image_path = "/Users/chanyoungko/Desktop/HandWriting/analyze_image/스크린샷 2025-04-28 오후 9.28.06.png"
    main(image_path)