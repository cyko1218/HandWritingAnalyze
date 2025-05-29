import pandas as pd
import numpy as np
import os


def create_balanced_dataset(input_csv_path, output_csv_path):
    """
    CSV 파일에서 라벨 0과 1이 동일한 비율로 있는 균형 잡힌 데이터셋 생성

    Args:
        input_csv_path: 입력 CSV 파일 경로
        output_csv_path: 출력 CSV 파일 경로
    """
    print(f"CSV 파일 로드 중: {input_csv_path}")

    # CSV 파일 로드
    try:
        df = pd.read_csv(input_csv_path)
    except Exception as e:
        print(f"CSV 파일 로드 중 오류 발생: {e}")
        # 파일 경로가 너무 길어서 문제가 있을 수 있으므로 직접 처리
        with open(input_csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        headers = lines[0].strip().split(',')
        data = []
        for line in lines[1:]:
            # csv 파일의 경우 이미지 경로에 쉼표가 없다고 가정하고 마지막 쉼표로 구분
            parts = line.strip().rsplit(',', 1)
            if len(parts) == 2:
                img_paths = parts[0].split(',', 1)
                if len(img_paths) == 2:
                    img_path1, img_path2 = img_paths
                    label = parts[1]
                    data.append([img_path1, img_path2, label])
                else:
                    print(f"잘못된 라인 형식: {line}")
            else:
                print(f"잘못된 라인 형식: {line}")

        df = pd.DataFrame(data, columns=headers)

    # 데이터 분포 확인
    label_counts = df['label'].value_counts()
    print(f"\n원본 데이터 분포:")
    print(f"라벨 0 (다른 작성자): {label_counts.get(0, 0)}개")
    print(f"라벨 1 (같은 작성자): {label_counts.get(1, 0)}개")

    # 라벨별로 데이터 분리
    label_0_df = df[df['label'] == 0]
    label_1_df = df[df['label'] == 1]

    # 더 작은 개수 결정
    min_count = min(len(label_0_df), len(label_1_df))
    print(f"\n각 라벨당 선택할 샘플 수: {min_count}개")

    # 균형 있게 샘플링
    if min_count > 0:
        balanced_0 = label_0_df.sample(min_count, random_state=42)
        balanced_1 = label_1_df.sample(min_count, random_state=42)

        # 데이터 합치기
        balanced_df = pd.concat([balanced_0, balanced_1])

        # 데이터 섞기
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # 저장
        balanced_df.to_csv(output_csv_path, index=False)
        print(f"\n균형 잡힌 데이터셋 생성 완료:")
        print(f"라벨 0 (다른 작성자): {len(balanced_df[balanced_df['label'] == 0])}개")
        print(f"라벨 1 (같은 작성자): {len(balanced_df[balanced_df['label'] == 1])}개")
        print(f"전체 샘플 수: {len(balanced_df)}개")
        print(f"저장 경로: {output_csv_path}")
    else:
        print("\n오류: 한 라벨의 샘플 수가 0입니다. 균형 잡힌 데이터셋을 생성할 수 없습니다.")


def verify_image_paths(csv_path):
    """
    CSV 파일에 있는 이미지 경로가 실제로 존재하는지 확인

    Args:
        csv_path: CSV 파일 경로

    Returns:
        bool: 모든 이미지가 존재하면 True, 아니면 False
    """
    print(f"\n이미지 경로 검증 중...")
    df = pd.read_csv(csv_path)

    # 열 이름 확인
    img1_col = df.columns[0]
    img2_col = df.columns[1]

    all_valid = True
    missing_paths = []

    for i, row in df.iterrows():
        img1_path = row[img1_col]
        img2_path = row[img2_col]

        if not os.path.exists(img1_path):
            missing_paths.append(img1_path)
            all_valid = False

        if not os.path.exists(img2_path):
            missing_paths.append(img2_path)
            all_valid = False

    if all_valid:
        print("✅ 모든 이미지 경로가 유효합니다.")
    else:
        print(f"⚠️ {len(missing_paths)}개의 이미지를 찾을 수 없습니다.")
        if len(missing_paths) < 10:
            for path in missing_paths:
                print(f"   - {path}")
        else:
            print(f"   - {missing_paths[0]}")
            print(f"   - {missing_paths[1]}")
            print(f"   - {missing_paths[2]}")
            print(f"   - ... 외 {len(missing_paths) - 3}개")

    return all_valid


if __name__ == "__main__":
    # 입력 및 출력 CSV 파일 경로 설정
    input_csv = "/Users/chanyoungko/Desktop/HandWriting/handwriting_pairs_train.csv"  # 원본 CSV 파일
    output_csv = "handwriting_balanced.csv"  # 균형 잡힌 CSV 파일

    print("=" * 60)
    print("필기체 데이터 균형 맞추기")
    print("=" * 60)

    # 사용자로부터 파일 경로 입력 받기 (선택 사항)
    custom_input = input("입력 CSV 파일 경로 (기본값 사용: Enter): ")
    if custom_input.strip():
        input_csv = custom_input

    custom_output = input("출력 CSV 파일 경로 (기본값 사용: Enter): ")
    if custom_output.strip():
        output_csv = custom_output

    # 균형 잡힌 데이터셋 생성
    create_balanced_dataset(input_csv, output_csv)

    # 이미지 경로 검증 여부 확인
    verify_option = input("\n생성된 CSV 파일의 이미지 경로를 검증하시겠습니까? (y/n): ")
    if verify_option.lower() == 'y':
        verify_image_paths(output_csv)

    print("\n작업 완료!")