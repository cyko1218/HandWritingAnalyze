import os
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# 한글 폰트 설정 함수
def set_korean_font():
    try:
        # Windows 환경
        if os.name == 'nt':
            font_path = 'c:/Windows/Fonts/malgun.ttf'  # 맑은 고딕 폰트 사용
            if not os.path.exists(font_path):
                # 다른 한글 폰트 시도
                font_path = 'c:/Windows/Fonts/gulim.ttc'  # 굴림체 시도
        # macOS 환경 (맥 환경으로 보이므로 이 부분이 중요합니다)
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



class HandwritingPersonalityAnalyzer:
    def __init__(self):
        """필기체 성격 분석기 초기화"""
        # 임계값 설정 (실제 데이터 분석 후 조정 필요)
        self.thresholds = {
            'size': {'small': 500, 'large': 2000},
            'roundness': {'angular': 0.5},
            'pressure': {'high': 0.7},
            'tilt': {'right': 5, 'left': -5},
            'connectivity': {'connected': 0.6},
            'spacing': {'wide': 30, 'narrow': 10},
            'regularity': {'regular': 0.2},
            'speed': {'fast': 0.7}
        }

        # 성격 특성 매핑
        self.personality_traits = {
            'size': {
                'small': "절약 정신, 보수적, 공손함, 치밀함, 내향적, 조심스러움",
                'medium': "균형 잡힌 성격",
                'large': "용기와 사회성 있음, 낭비적 성향, 외향적, 말이 많고 표현하는 것을 즐김"
            },
            'shape': {
                'angular': "규범을 잘 지킴, 정직함, 고집스러움, 원칙을 중시, 융통성 없음",
                'round': "성격이 밝고 원만함, 합리적임, 상상력이 풍부, 아이디어가 많음, 사고가 유연함"
            },
            'pressure': {
                'strong': "정신력이 강함, 의지가 굳음, 활력이 있음, 자기주장이 강함, 호전적임",
                'weak': "에너지가 약함, 복종, 유순함, 수줍음"
            },
            'tilt': {
                'right_up': "낙관적, 열정적, 희망적",
                'right_down': "차가움, 감정표현을 잘 안함, 비관적, 비판적",
                'neutral': "균형 잡힌 감정, 안정적"
            },
            'connectivity': {
                'connected': "논리적, 합리적, 사물의 연결이나 사람과의 관계를 이해함",
                'disconnected': "직관적, 감각적, 사물의 연결이나 사람과의 관계에 다소 냉담함"
            },
            'spacing': {
                'wide': "포용력 있음, 상대방의 말을 잘 들어줌, 새로운 지식과 정보를 적극 수용함",
                'narrow': "남의 말을 잘 수용하지 않음, 착실함, 한가지를 파고듦",
                'medium': "균형 있는 사고방식"
            },
            'line_spacing': {
                'wide': "조심스러움, 사려깊음, 남에게 피해 주는 것을 싫어함",
                'narrow': "활력이 있음, 조심스럽지 못함, 경솔함, 남을 배려하지 않음",
                'medium': "적절한 사회성과 배려심"
            },
            'regularity': {
                'regular': "일관성, 안정 지향적, 신뢰할 만함, 유연성 부족",
                'irregular': "활력, 자유분방, 즉흥적, 충동적, 기분파, 인내력이 약함"
            },
            'speed': {
                'fast': "민첩함, 활발함, 변화 욕구, 기백이 있음, 열정적, 변덕스러움, 경솔함",
                'slow': "느긋함, 용의주도, 신중, 끈기, 우유부단"
            }
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
        """필적의 특성 추출 함수"""
        try:
            binary = processed_images['binary']
            gray = processed_images['gray']
            features = {}

            # 1. 글씨 크기 분석
            contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            char_sizes = [cv2.contourArea(cnt) for cnt in contours if cv2.contourArea(cnt) > 10]

            if char_sizes:
                features['avg_size'] = np.mean(char_sizes)
                features['size_std'] = np.std(char_sizes)
            else:
                features['avg_size'] = 0
                features['size_std'] = 0

            # 2. 글씨 모양 분석 (각진 정도 vs 둥근 정도)
            roundness_values = []
            for cnt in contours:
                if cv2.contourArea(cnt) > 10:
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter > 0:
                        area = cv2.contourArea(cnt)
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        roundness_values.append(circularity)

            features['roundness'] = np.mean(roundness_values) if roundness_values else 0

            # 3. 필압 분석 (픽셀 강도의 변화)
            # 이진화된 이미지에서는 정확한 필압을 측정하기 어려움
            # 대안으로, 원본 그레이스케일 이미지에서 필기 부분의 강도 분석
            if np.max(binary) > 0:
                mask = binary > 0
                if np.sum(mask) > 0:
                    inverted_gray = 255 - gray  # 흰 배경, 검은 글씨를 검은 배경, 흰 글씨로 반전
                    pressure_values = inverted_gray[mask]
                    features['pressure'] = np.mean(pressure_values) / 255.0  # 0~1 사이로 정규화
                else:
                    features['pressure'] = 0
            else:
                features['pressure'] = 0

            # 4. 기울기 분석
            # 히스토그램 프로젝션 방법으로 기울기 추정
            projection = []
            for angle in range(-45, 46, 5):  # -45도부터 45도까지 5도 간격
                rotated = self._rotate_image(binary, angle)
                proj = np.sum(rotated, axis=1)  # 수평 프로젝션
                projection.append(np.std(proj))  # 프로젝션의 표준편차

            if projection:
                max_idx = np.argmax(projection)
                features['tilt_angle'] = -45 + (max_idx * 5)
            else:
                features['tilt_angle'] = 0

            # 5. 획의 연결성 분석
            # 연결 요소 레이블링
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
            if num_labels > 1:  # 배경을 제외한 연결 요소
                # 획 당 평균 픽셀 수
                stroke_pixels = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
                features['avg_stroke_length'] = np.mean(stroke_pixels) if stroke_pixels else 0

                # 연결된 획의 비율 (50픽셀 이상인 획을 연결된 것으로 간주)
                connected_strokes = sum(1 for s in stroke_pixels if s > 50)
                features['stroke_connectivity'] = connected_strokes / (num_labels - 1) if num_labels > 1 else 0
            else:
                features['avg_stroke_length'] = 0
                features['stroke_connectivity'] = 0

            # 6. 획 사이 공간 및 글자 간격 분석
            # 수평/수직 투영 프로필로 간격 측정
            h_proj = np.sum(binary, axis=1)
            v_proj = np.sum(binary, axis=0)

            # 빈 공간 감지 (값이 0인 위치)
            h_spaces = np.where(h_proj == 0)[0]
            v_spaces = np.where(v_proj == 0)[0]

            # 연속된 빈 공간의 길이 계산
            h_space_lengths = []
            v_space_lengths = []

            if len(h_spaces) > 0:
                # 연속된 인덱스 그룹으로 분할
                h_groups = np.split(h_spaces, np.where(np.diff(h_spaces) != 1)[0] + 1)
                h_space_lengths = [len(g) for g in h_groups if len(g) > 0]

            if len(v_spaces) > 0:
                v_groups = np.split(v_spaces, np.where(np.diff(v_spaces) != 1)[0] + 1)
                v_space_lengths = [len(g) for g in v_groups if len(g) > 0]

            # 평균 공간 크기
            features['avg_h_space'] = np.mean(h_space_lengths) if h_space_lengths else 0
            features['avg_v_space'] = np.mean(v_space_lengths) if v_space_lengths else 0

            # 7. 규칙성 분석
            # 글자 크기의 변동 계수
            features['size_cv'] = (np.std(char_sizes) / np.mean(char_sizes)
                                   if char_sizes and np.mean(char_sizes) > 0 else 0)

            # 글자 간격의 변동 계수
            features['space_cv'] = (np.std(v_space_lengths) / np.mean(v_space_lengths)
                                    if v_space_lengths and np.mean(v_space_lengths) > 0 else 0)

            # 8. 텍스처 특성 (필기 속도와 관련된 특성)
            if np.max(binary) > 0:
                # GLCM 계산 (Gray-Level Co-occurrence Matrix)
                glcm = graycomatrix(binary, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256,
                                    symmetric=True, normed=True)

                # GLCM 특성 추출
                features['contrast'] = np.mean(graycoprops(glcm, 'contrast'))
                features['dissimilarity'] = np.mean(graycoprops(glcm, 'dissimilarity'))
                features['homogeneity'] = np.mean(graycoprops(glcm, 'homogeneity'))
                features['energy'] = np.mean(graycoprops(glcm, 'energy'))
                features['correlation'] = np.mean(graycoprops(glcm, 'correlation'))
            else:
                features['contrast'] = 0
                features['dissimilarity'] = 0
                features['homogeneity'] = 0
                features['energy'] = 0
                features['correlation'] = 0

            return features

        except Exception as e:
            print(f"특징 추출 중 오류 발생: {str(e)}")
            return None

    def analyze_personality(self, features):
        """추출된 특성을 바탕으로 성격 분석"""
        try:
            personality = {}

            # 1. 글씨 크기에 따른 성격
            avg_size = features['avg_size']
            if avg_size < self.thresholds['size']['small']:
                personality['size'] = {'category': 'small', 'traits': self.personality_traits['size']['small']}
            elif avg_size > self.thresholds['size']['large']:
                personality['size'] = {'category': 'large', 'traits': self.personality_traits['size']['large']}
            else:
                personality['size'] = {'category': 'medium', 'traits': self.personality_traits['size']['medium']}

            # 2. 글씨 모양에 따른 성격
            roundness = features['roundness']
            if roundness < self.thresholds['roundness']['angular']:
                personality['shape'] = {'category': 'angular', 'traits': self.personality_traits['shape']['angular']}
            else:
                personality['shape'] = {'category': 'round', 'traits': self.personality_traits['shape']['round']}

            # 3. 필압에 따른 성격
            pressure = features['pressure']
            if pressure > self.thresholds['pressure']['high']:
                personality['pressure'] = {'category': 'strong',
                                           'traits': self.personality_traits['pressure']['strong']}
            else:
                personality['pressure'] = {'category': 'weak', 'traits': self.personality_traits['pressure']['weak']}

            # 4. 기울기에 따른 성격
            tilt = features['tilt_angle']
            if tilt > self.thresholds['tilt']['right']:
                personality['tilt'] = {'category': 'right_up', 'traits': self.personality_traits['tilt']['right_up']}
            elif tilt < self.thresholds['tilt']['left']:
                personality['tilt'] = {'category': 'right_down',
                                       'traits': self.personality_traits['tilt']['right_down']}
            else:
                personality['tilt'] = {'category': 'neutral', 'traits': self.personality_traits['tilt']['neutral']}

            # 5. 획의 연결성에 따른 성격
            connectivity = features['stroke_connectivity']
            if connectivity > self.thresholds['connectivity']['connected']:
                personality['connectivity'] = {'category': 'connected',
                                               'traits': self.personality_traits['connectivity']['connected']}
            else:
                personality['connectivity'] = {'category': 'disconnected',
                                               'traits': self.personality_traits['connectivity']['disconnected']}

            # 6. 글자 간격에 따른 성격
            avg_v_space = features['avg_v_space']
            if avg_v_space > self.thresholds['spacing']['wide']:
                personality['char_spacing'] = {'category': 'wide', 'traits': self.personality_traits['spacing']['wide']}
            elif avg_v_space < self.thresholds['spacing']['narrow']:
                personality['char_spacing'] = {'category': 'narrow',
                                               'traits': self.personality_traits['spacing']['narrow']}
            else:
                personality['char_spacing'] = {'category': 'medium',
                                               'traits': self.personality_traits['spacing']['medium']}

            # 7. 행간 간격에 따른 성격
            avg_h_space = features['avg_h_space']
            if avg_h_space > self.thresholds['spacing']['wide']:
                personality['line_spacing'] = {'category': 'wide',
                                               'traits': self.personality_traits['line_spacing']['wide']}
            elif avg_h_space < self.thresholds['spacing']['narrow']:
                personality['line_spacing'] = {'category': 'narrow',
                                               'traits': self.personality_traits['line_spacing']['narrow']}
            else:
                personality['line_spacing'] = {'category': 'medium',
                                               'traits': self.personality_traits['line_spacing']['medium']}

            # 8. 규칙성에 따른 성격
            regularity = features['size_cv']  # 크기의 변동 계수로 규칙성 판단
            if regularity < self.thresholds['regularity']['regular']:
                personality['regularity'] = {'category': 'regular',
                                             'traits': self.personality_traits['regularity']['regular']}
            else:
                personality['regularity'] = {'category': 'irregular',
                                             'traits': self.personality_traits['regularity']['irregular']}

            # 9. 속도에 따른 성격 (GLCM 특성으로 추정)
            speed_indicator = features['contrast']  # 높은 대비는 빠른 필기와 관련
            if speed_indicator > self.thresholds['speed']['fast']:
                personality['speed'] = {'category': 'fast', 'traits': self.personality_traits['speed']['fast']}
            else:
                personality['speed'] = {'category': 'slow', 'traits': self.personality_traits['speed']['slow']}

            return personality

        except Exception as e:
            print(f"성격 분석 중 오류 발생: {str(e)}")
            return None

    def generate_report(self, personality):
        """성격 분석 결과를 보고서 형태로 생성"""
        report = {
            'summary': self._generate_summary(personality),
            'details': personality,
            'dominant_traits': self._extract_dominant_traits(personality)
        }
        return report

    def _generate_summary(self, personality):
        """성격 분석 결과의 요약 생성"""
        try:
            # 대략적인 성격 유형 추론
            intro_extro = "내향적" if personality['size']['category'] == 'small' else "외향적"
            rational_emotional = "이성적" if personality['shape']['category'] == 'angular' else "감성적"
            cautious_bold = "신중한" if personality['speed']['category'] == 'slow' else "대담한"

            summary = f"{intro_extro}이고 {rational_emotional}이며 {cautious_bold} 성격의 소유자입니다. "

            # 주요 특성 추가
            if personality['pressure']['category'] == 'strong':
                summary += "강한 의지력과 자기주장이 있으며, "
            else:
                summary += "유순하고 감성적이며, "

            if personality['tilt']['category'] == 'right_up':
                summary += "낙관적이고 열정적인 성향을 보입니다. "
            elif personality['tilt']['category'] == 'right_down':
                summary += "비판적이고 분석적인 성향을 보입니다. "
            else:
                summary += "균형 잡힌 사고방식을 가지고 있습니다. "

            # 사회적 특성 추가
            if personality['char_spacing']['category'] == 'wide':
                summary += "다른 사람의 의견을 존중하고 새로운 정보를 수용하는 편입니다."
            else:
                summary += "자신의 의견을 중요시하며 한 가지에 깊이 집중하는 편입니다."

            return summary

        except Exception as e:
            print(f"요약 생성 중 오류 발생: {str(e)}")
            return "성격 특성을 요약하는 과정에서 오류가 발생했습니다."

    def _extract_dominant_traits(self, personality):
        """가장 두드러진 성격 특성 추출"""
        traits = []

        # 모든 특성에서 중요한 키워드 추출
        for category, data in personality.items():
            traits_text = data['traits']
            traits_list = [trait.strip() for trait in traits_text.split(',')]
            traits.extend(traits_list)

        # 중복 제거 및 빈도수 계산
        trait_counts = {}
        for trait in traits:
            trait_counts[trait] = trait_counts.get(trait, 0) + 1

        # 가장 빈번한 특성 5개 추출
        dominant_traits = sorted(trait_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        return [trait for trait, count in dominant_traits]

    def _rotate_image(self, image, angle):
        """이미지를 주어진 각도로 회전"""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)
        return rotated

    def visualize_analysis(self, processed_images, features, personality):
        """분석 결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 원본 이미지
        axes[0, 0].imshow(cv2.cvtColor(processed_images['original'], cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('원본 이미지')
        axes[0, 0].axis('off')

        # 이진화 이미지
        axes[0, 1].imshow(processed_images['binary'], cmap='gray')
        axes[0, 1].set_title('이진화 이미지')
        axes[0, 1].axis('off')

        # 특성 시각화 (수평 막대 그래프)
        feature_labels = ['글씨 크기', '둥근 정도', '필압', '기울기', '연결성', '글자 간격', '행간', '규칙성', '속도']
        feature_values = [
            features['avg_size'] / self.thresholds['size']['large'],
            features['roundness'],
            features['pressure'],
            (features['tilt_angle'] + 45) / 90,  # -45~45 -> 0~1
            features['stroke_connectivity'],
            features['avg_v_space'] / self.thresholds['spacing']['wide'],
            features['avg_h_space'] / self.thresholds['spacing']['wide'],
            1 - features['size_cv'],  # 변동계수가 작을수록 규칙적
            features['contrast'] / self.thresholds['speed']['fast']
        ]

        # 값을 0~1 범위로 클리핑
        feature_values = [max(0, min(1, val)) for val in feature_values]

        # 수평 막대 그래프
        y_pos = np.arange(len(feature_labels))
        axes[1, 0].barh(y_pos, feature_values, align='center')
        axes[1, 0].set_yticks(y_pos)
        axes[1, 0].set_yticklabels(feature_labels)
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_title('필적 특성')

        # 성격 특성 텍스트
        axes[1, 1].axis('off')
        summary_text = personality['summary']
        dominant_traits = personality['dominant_traits']

        text = f"성격 특성 요약:\n{summary_text}\n\n"
        text += "주요 특성:\n"
        for i, trait in enumerate(dominant_traits, 1):
            text += f"{i}. {trait}\n"

        axes[1, 1].text(0, 0.5, text, fontsize=12, va='center', wrap=True)
        axes[1, 1].set_title('성격 분석 결과')

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
        personality_traits = self.analyze_personality(features)
        if personality_traits is None:
            return {"error": "성격 분석 실패"}

        # 4. 결과 보고서 생성
        report = self.generate_report(personality_traits)

        # 5. 결과 시각화 (선택적)
        if visualize:
            visualization = self.visualize_analysis(processed_images, features, report)
            report['visualization'] = visualization

        return {
            "features": features,
            "personality": report
        }


# 앱 사용 예시
def main():
    analyzer = HandwritingPersonalityAnalyzer()

    # 테스트 이미지 경로
    image_path = "/Users/chanyoungko/Desktop/HandWriting/analyze_image/스크린샷 2025-04-28 오후 9.28.06.png"  # 실제 이미지 경로로 변경 필요

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
    print("\n요약:")
    print(result['personality']['summary'])

    print("\n주요 성격 특성:")
    for i, trait in enumerate(result['personality']['dominant_traits'], 1):
        print(f"{i}. {trait}")

    print("\n상세 분석:")
    for category, data in result['personality']['details'].items():
        if category != 'summary' and category != 'dominant_traits':
            print(f"- {category}: {data['category']} ({data['traits']})")

    # 시각화 결과 표시
    if 'visualization' in result['personality']:
        plt.show()


if __name__ == "__main__":
    # 한글 폰트 설정 호출
    set_korean_font()
    analyzer = HandwritingPersonalityAnalyzer()


    main()