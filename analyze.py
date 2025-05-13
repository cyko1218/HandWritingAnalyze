import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label  # measurements.label 대신 label 직접 import
from collections import Counter
import matplotlib.font_manager as fm
import platform


# 한글 폰트 설정
def set_korean_font():
    system = platform.system()
    if system == 'Darwin':  # macOS
        plt.rc('font', family='AppleGothic')
    elif system == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    elif system == 'Linux':
        plt.rc('font', family='NanumGothic')
    else:
        # 기본 폰트 설정
        plt.rc('font', family='DejaVu Sans')

    # 음수 표시 문제 해결
    plt.rcParams['axes.unicode_minus'] = False


# 한글 폰트 설정 적용
set_korean_font()


class HandwritingAnalyzer:
    def __init__(self, image_path):
        """초기화 함수"""
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError("이미지를 불러올 수 없습니다.")

        # 이미지 전처리
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        _, self.binary_image = cv2.threshold(self.gray_image, 150, 255, cv2.THRESH_BINARY_INV)

        # 결과 저장 변수 초기화
        self.line_gaps = []  # 행간 간격
        self.pressure_level = 0  # 필압 수준
        self.slant_angle = 0  # 기울기 각도
        self.character_sizes = []  # 글자 크기
        self.character_spacing = []  # 글자 간격
        self.results = {}  # 분석 결과

    def analyze(self):
        """필기 분석 실행"""
        self._analyze_line_gaps()
        self._analyze_pressure()
        self._analyze_character_size()
        self._analyze_slant()
        self._analyze_regularity()
        self._analyze_character_spacing()
        self._analyze_closed_shapes()

        # 결과 해석
        self._interpret_results()

        return self.results

    def _analyze_line_gaps(self):
        """행간 분석"""
        # 수평 프로젝션 프로필 계산
        h_proj = np.sum(self.binary_image, axis=1)

        # 줄 찾기
        lines = []
        in_line = False
        line_start = 0

        for i, proj in enumerate(h_proj):
            if not in_line and proj > 0:
                in_line = True
                line_start = i
            elif in_line and proj == 0:
                in_line = False
                lines.append((line_start, i))

        # 행간 계산
        if len(lines) > 1:
            for i in range(len(lines) - 1):
                gap = lines[i + 1][0] - lines[i][1]
                if gap > 0:  # 음수 간격은 제외
                    self.line_gaps.append(gap)

        # 행간 평균 계산
        if self.line_gaps:
            avg_gap = sum(self.line_gaps) / len(self.line_gaps)
            self.results['line_gap'] = avg_gap

            # 행간 좁은지 판단
            img_height = self.binary_image.shape[0]
            if avg_gap < img_height * 0.03:  # 이미지 높이의 3% 미만이면 좁다고 판단
                self.results['narrow_line_gaps'] = True
            else:
                self.results['narrow_line_gaps'] = False
        else:
            self.results['line_gap'] = 0
            self.results['narrow_line_gaps'] = False

    def _analyze_pressure(self):
        """필압 분석"""
        # 이진화된 이미지에서 검은색 픽셀의 평균 강도를 계산
        # 원본 흑백 이미지에서 글자 부분(이진화 이미지에서 255)의 평균 강도 계산
        mask = (self.binary_image == 255)
        if np.any(mask):
            stroke_intensity = 255 - np.mean(self.gray_image[mask])
            # 0-100 스케일로 정규화
            self.pressure_level = int((stroke_intensity / 255) * 100)
            self.results['pressure'] = self.pressure_level

            if self.pressure_level > 70:
                self.results['strong_pressure'] = True
            else:
                self.results['strong_pressure'] = False
        else:
            self.results['pressure'] = 0
            self.results['strong_pressure'] = False

    def _analyze_character_size(self):
        """글자 크기 분석"""
        # 연결 요소 레이블링 - 수정된 부분
        labeled, num_features = label(self.binary_image)

        if num_features > 0:
            # 각 연결 요소의 크기 계산
            for i in range(1, num_features + 1):
                component = (labeled == i)
                y, x = np.where(component)

                if len(y) > 10:  # 노이즈 제거
                    height = max(y) - min(y)
                    width = max(x) - min(x)
                    size = height * width
                    self.character_sizes.append(size)

            if self.character_sizes:
                avg_size = sum(self.character_sizes) / len(self.character_sizes)
                self.results['character_size'] = avg_size

                # 글자 크기 변동성 계산
                if len(self.character_sizes) > 1:
                    size_std = np.std(self.character_sizes)
                    size_variation = size_std / avg_size
                    self.results['size_regularity'] = 1 - min(size_variation, 1)  # 0-1 사이 값, 1이 가장 규칙적
                else:
                    self.results['size_regularity'] = 1

                # 글자 크기 해석
                img_area = self.binary_image.shape[0] * self.binary_image.shape[1]
                relative_size = avg_size / img_area

                if relative_size < 0.001:
                    self.results['small_characters'] = True
                else:
                    self.results['small_characters'] = False
            else:
                self.results['character_size'] = 0
                self.results['size_regularity'] = 0
                self.results['small_characters'] = False
        else:
            self.results['character_size'] = 0
            self.results['size_regularity'] = 0
            self.results['small_characters'] = False

    def _analyze_slant(self):
        """글자 기울기 분석"""
        # 허프 변환을 사용하여 직선 감지
        edges = cv2.Canny(self.binary_image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 50)

        angles = []
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                # 수직에 가까운 선만 고려 (글자의 수직 획)
                if 0.5 < theta < 2.5:  # 약 30도 ~ 150도
                    angle = 90 - np.degrees(theta)  # 수직(90도)에서의 편차
                    angles.append(angle)

        if angles:
            # 중앙값 기울기 계산 (이상치 영향 줄이기)
            self.slant_angle = np.median(angles)
            self.results['slant'] = self.slant_angle

            # 기울기 해석
            if -15 < self.slant_angle < 15:
                self.results['slant_type'] = "수직"
            elif self.slant_angle >= 15:
                self.results['slant_type'] = "오른쪽"
            else:
                self.results['slant_type'] = "왼쪽"
        else:
            self.results['slant'] = 0
            self.results['slant_type'] = "수직"

    def _analyze_regularity(self):
        """글자 규칙성 분석"""
        # 이미 계산된 크기 변동성을 사용
        if 'size_regularity' in self.results:
            size_reg = self.results['size_regularity']
        else:
            size_reg = 0

        # 기울기 일관성 계산
        if hasattr(self, 'slant_angles') and len(self.slant_angles) > 1:
            slant_std = np.std(self.slant_angles)
            slant_reg = 1 - min(slant_std / 45, 1)  # 0-1 사이 값
        else:
            slant_reg = 1

        # 종합 규칙성 점수 (0-1, 1이 가장 규칙적)
        regularity = (size_reg + slant_reg) / 2
        self.results['regularity'] = regularity

        # 불규칙성 해석
        if regularity < 0.4:
            self.results['irregular_writing'] = True
        else:
            self.results['irregular_writing'] = False

    def _analyze_character_spacing(self):
        """글자 간격 분석"""
        # 연결 요소 레이블링 - 수정된 부분
        labeled, num_features = label(self.binary_image)

        # 각 연결 요소의 중심점 계산
        centers = []
        for i in range(1, num_features + 1):
            component = (labeled == i)
            y, x = np.where(component)

            if len(y) > 10:  # 노이즈 제거
                center_y = np.mean(y)
                center_x = np.mean(x)
                centers.append((center_x, center_y))

        # 같은 줄에 있는 인접 글자 간 간격 계산
        if len(centers) > 1:
            # 중심점을 x 좌표로 정렬
            centers.sort(key=lambda c: c[0])

            # y 좌표가 비슷한 글자들 그룹핑
            y_threshold = self.binary_image.shape[0] * 0.05
            lines = []
            current_line = [centers[0]]

            for i in range(1, len(centers)):
                if abs(centers[i][1] - current_line[0][1]) < y_threshold:
                    current_line.append(centers[i])
                else:
                    if len(current_line) > 1:
                        lines.append(current_line)
                    current_line = [centers[i]]

            if len(current_line) > 1:
                lines.append(current_line)

            # 각 줄에서 인접 글자 간 간격 계산
            spacings = []
            for line in lines:
                for i in range(len(line) - 1):
                    spacing = line[i + 1][0] - line[i][0]
                    spacings.append(spacing)

            if spacings:
                avg_spacing = sum(spacings) / len(spacings)
                self.results['character_spacing'] = avg_spacing

                # 이미지 너비에 상대적인 간격 계산
                img_width = self.binary_image.shape[1]
                relative_spacing = avg_spacing / img_width

                if relative_spacing > 0.05:
                    self.results['wide_spacing'] = True
                else:
                    self.results['wide_spacing'] = False
            else:
                self.results['character_spacing'] = 0
                self.results['wide_spacing'] = False
        else:
            self.results['character_spacing'] = 0
            self.results['wide_spacing'] = False

    def _analyze_closed_shapes(self):
        """닫힌 모양(예: ㅁ, ㅇ 등) 분석"""
        # 윤곽선 감지
        contours, _ = cv2.findContours(self.binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        closed_shapes = []
        for contour in contours:
            # 윤곽선이 충분히 크고 닫혀있는지 확인
            area = cv2.contourArea(contour)
            if area > 50:  # 작은 노이즈 제거
                # 닫힌 형태 근사
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # 모양이 닫혀있고 사각형에 가까운지 확인
                if len(approx) >= 4 and cv2.isContourConvex(approx):
                    x, y, w, h = cv2.boundingRect(contour)

                    # 아래쪽이 닫혀있는지 확인 (하단의 검은 픽셀 밀도)
                    bottom_region = self.binary_image[y + h - 3:y + h + 1, x:x + w]
                    if bottom_region.size > 0:
                        bottom_density = np.sum(bottom_region) / bottom_region.size

                        if bottom_density > 0.5:  # 하단이 충분히 닫혀있음
                            closed_shapes.append("closed_bottom")

        # ㅁ 형태의 하단이 닫힌 모양 개수
        self.results['closed_bottom_shapes'] = len(closed_shapes)
        if len(closed_shapes) > 0:
            self.results['has_closed_bottom'] = True
        else:
            self.results['has_closed_bottom'] = False

    def _interpret_results(self):
        """분석 결과 해석"""
        personality_traits = []

        # 행간 분석
        if self.results.get('narrow_line_gaps', False):
            personality_traits.append("- 좁은 행간과 서로 침범하는 글씨는 남에게 피해주는 걸 개의치 않는 성향입니다.")
        else:
            personality_traits.append("- 행간이 넓어 조심스럽고 사려깊은 성향입니다.")

        # 필압 분석
        if self.results.get('strong_pressure', False):
            personality_traits.append("- 강한 필압은 육체적 에너지가 강하고 감정 표현이 직접적임을 나타냅니다.")
        else:
            personality_traits.append("- 보통 혹은 약한 필압은 섬세하고 감정 조절이 잘 됨을 나타냅니다.")

        # 규칙성 분석
        if self.results.get('irregular_writing', False):
            personality_traits.append("- 불규칙적인 글씨는 충동성이 강하고 감정 변화가 큰 성향입니다.")
        else:
            personality_traits.append("- 규칙적인 글씨는 안정적이고 체계적인 성향을 나타냅니다.")

        # 글자 크기 분석
        if self.results.get('small_characters', False):
            personality_traits.append("- 글씨 크기가 작아 보수적이고 겸손한 성향입니다.")
        else:
            personality_traits.append("- 글씨 크기가 보통이거나 큰 편으로 자신감이 있고 사교적인 성향입니다.")

        # 글자 크기 규칙성
        if self.results.get('size_regularity', 0) > 0.7:
            personality_traits.append("- 글씨 크기가 규칙적이어서 의지가 강하고 집중력이 좋습니다.")

        # 기울기 분석
        slant_type = self.results.get('slant_type', "수직")
        if slant_type == "오른쪽":
            personality_traits.append("- 오른쪽으로 기울어진 글씨는 감정적이고 사교적인 성향입니다.")
        elif slant_type == "왼쪽":
            personality_traits.append("- 왼쪽으로 기울어진 글씨는 자기 보호적이고 방어적인 성향입니다.")
        else:
            personality_traits.append("- 수직의 글씨는 논리적이고 이성적인 성향입니다.")

        # 각진 글씨 분석
        if self.results.get('angular_writing', True):  # 기본값 설정
            personality_traits.append("- 글씨가 각이 져 있어 신중하고 이성적인 성향입니다.")

        # 닫힌 모양 분석
        if self.results.get('has_closed_bottom', False):
            personality_traits.append("- 하단이 굳게 닫힌 ㅁ이 있어 절약형 성향입니다.")

        self.results['personality_traits'] = personality_traits

    def generate_report(self):
        """분석 결과 보고서 생성"""
        if not self.results:
            self.analyze()

        report = "=== 필기체 성격 분석 보고서 ===\n\n"

        # 수치 정보 출력
        report += "## 측정 수치 정보\n"
        report += f"- 필압 수준: {self.results.get('pressure', 0)}/100\n"
        report += f"- 글씨 기울기: {self.results.get('slant', 0):.1f}도\n"
        report += f"- 행간 간격: {self.results.get('line_gap', 0):.1f}픽셀\n"
        report += f"- 글자 간격: {self.results.get('character_spacing', 0):.1f}픽셀\n\n"

        # 성격 특성 출력
        report += "## 성격 특성 분석\n"
        for trait in self.results.get('personality_traits', []):
            report += f"{trait}\n"

        return report

    def visualize(self):
        """분석 결과 시각화"""
        plt.figure(figsize=(12, 8))

        # 원본 이미지
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        plt.title('원본 이미지')
        plt.axis('off')

        # 이진화 이미지
        plt.subplot(2, 2, 2)
        plt.imshow(self.binary_image, cmap='gray')
        plt.title('이진화 이미지')
        plt.axis('off')

        # 측정 수치 시각화
        plt.subplot(2, 2, 3)
        metrics = ['필압', '기울기', '행간', '글자간격']
        values = [
            self.results.get('pressure', 0) / 100,
            abs(self.results.get('slant', 0)) / 90,
            min(self.results.get('line_gap', 0) / 100, 1),
            min(self.results.get('character_spacing', 0) / 100, 1)
        ]

        plt.bar(metrics, values, color='skyblue')
        plt.ylim(0, 1)
        plt.title('필기 특성 측정 (정규화)')

        # 성격 특성 요약
        plt.subplot(2, 2, 4)
        plt.axis('off')
        traits_text = '\n'.join(self.results.get('personality_traits', [])[:5])
        plt.text(0.1, 0.5, traits_text, fontsize=9, verticalalignment='center')
        plt.title('주요 성격 특성')

        plt.tight_layout()
        plt.savefig('handwriting_analysis_result.png')
        plt.show()


# 사용 예시
if __name__ == "__main__":
    # 파일 경로를 실제 이미지 파일로 변경하세요
    analyzer = HandwritingAnalyzer("/Users/chanyoungko/Desktop/HandWriting/analyze_image/스크린샷 2025-05-02 오후 5.36.50.png")
    analyzer.analyze()

    # 결과 출력
    report = analyzer.generate_report()
    print(report)

    # 결과 시각화
    analyzer.visualize()