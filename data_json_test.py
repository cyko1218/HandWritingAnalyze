import os
import json
import csv
import glob
from collections import defaultdict


def collect_person_data(root_dir):
    """
    지정된 루트 디렉토리부터 모든 JSON 파일을 검색하여 person ID별로 파일 정보를 수집

    Args:
        root_dir: 검색을 시작할 루트 디렉토리 (예: '.../TL')

    Returns:
        person_data: person ID를 키로 가지는 딕셔너리, 각 ID에 해당하는 파일 정보 리스트를 값으로 함
    """
    person_data = defaultdict(list)

    # JSON 파일 모두 찾기
    json_files = glob.glob(os.path.join(root_dir, '**', '*.json'), recursive=True)

    print(f"{len(json_files)}개의 JSON 파일을 찾았습니다.")

    processed_files = 0
    error_files = 0

    # 각 JSON 파일에서 person ID 추출
    for json_file in json_files:
        try:
            # UTF-8 BOM 처리를 위해 utf-8-sig 인코딩 사용
            with open(json_file, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)

                # person 정보가 있는지 확인
                if 'person' in data and 'id' in data['person']:
                    person_id = data['person']['id']
                    person_sex = data['person'].get('sex', '')
                    person_age = data['person'].get('age_group', '')
                    person_posture = data['person'].get('posture', '')

                    # 파일 경로에서 폴더 구조 정보 추출
                    rel_path = os.path.relpath(json_file, root_dir)
                    folder_path = os.path.dirname(rel_path)

                    # person 데이터 저장
                    person_data[person_id].append({
                        'file_path': json_file,
                        'folder_path': folder_path,
                        'sex': person_sex,
                        'age_group': person_age,
                        'posture': person_posture
                    })

                    processed_files += 1

                    # 처리 상황 업데이트 (100개마다 출력)
                    if processed_files % 100 == 0:
                        print(f"{processed_files}/{len(json_files)} 파일 처리 완료...")
                else:
                    print(f"person 정보 없음: {json_file}")

        except Exception as e:
            error_files += 1
            print(f"파일 처리 중 오류 발생: {json_file}, 오류: {str(e)}")

    print(f"총 {processed_files}개 파일 성공적으로 처리, {error_files}개 파일 처리 실패")

    return person_data


def export_to_csv(person_data, output_file):
    """
    수집된 person 데이터를 CSV 파일로 내보내기

    Args:
        person_data: person ID를 키로 가지고 파일 정보 리스트를 값으로 가지는 딕셔너리
        output_file: 출력할 CSV 파일 경로
    """
    # 출력 디렉토리가 없으면 생성
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # CSV 헤더 작성
        writer.writerow(['Person ID', 'Sex', 'Age Group', 'Posture', 'File Count', 'Folder Paths', 'File Paths'])

        # 각 person ID에 대한 데이터 작성
        for person_id, files in person_data.items():
            if files:  # 파일이 있는 경우만
                # 첫 번째 파일에서 성별, 나이 그룹, 자세 정보 가져오기
                sex = files[0]['sex']
                age_group = files[0]['age_group']
                posture = files[0]['posture']

                # 폴더 경로 및 파일 경로 목록 생성
                folder_paths = '; '.join(set(file_info['folder_path'] for file_info in files))
                file_paths = '; '.join(file_info['file_path'] for file_info in files)

                # CSV에 행 추가
                writer.writerow([
                    person_id,
                    sex,
                    age_group,
                    posture,
                    len(files),
                    folder_paths,
                    file_paths
                ])

    print(f"CSV 파일 저장 완료: {output_file}")


def create_person_folders(person_data, output_dir):
    """
    각 person ID별로 폴더를 생성하고 해당 폴더에 관련 파일을 매핑하는 정보를 저장

    Args:
        person_data: person ID를 키로 가지고 파일 정보 리스트를 값으로 가지는 딕셔너리
        output_dir: 폴더를 생성할 기본 디렉토리
    """
    # 출력 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)

    for person_id, files in person_data.items():
        # person ID로 폴더 생성
        person_folder = os.path.join(output_dir, person_id)
        os.makedirs(person_folder, exist_ok=True)

        # 해당 폴더에 CSV 파일 생성
        info_csv = os.path.join(person_folder, f"{person_id}_files.csv")

        with open(info_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # CSV 헤더 작성
            writer.writerow(['File Path', 'Folder Path', 'Sex', 'Age Group', 'Posture'])

            # 각 파일 정보 작성
            for file_info in files:
                writer.writerow([
                    file_info['file_path'],
                    file_info['folder_path'],
                    file_info['sex'],
                    file_info['age_group'],
                    file_info['posture']
                ])

        print(f"Person ID: {person_id}, 파일 수: {len(files)}, 폴더 생성 완료: {person_folder}")


def analyze_person_data(person_data):
    """
    Person 데이터 분석 및 통계 출력

    Args:
        person_data: person ID를 키로 가지는 딕셔너리
    """
    # 사람별 파일 수 계산
    file_counts = {person_id: len(files) for person_id, files in person_data.items()}

    # 통계 계산
    total_people = len(file_counts)
    total_files = sum(file_counts.values())
    min_files = min(file_counts.values()) if file_counts else 0
    max_files = max(file_counts.values()) if file_counts else 0
    avg_files = total_files / total_people if total_people > 0 else 0

    # 파일 수별 사람 수 분포
    file_count_distribution = defaultdict(int)
    for count in file_counts.values():
        file_count_distribution[count] += 1

    # 성별 통계
    gender_counts = defaultdict(int)
    for files in person_data.values():
        if files:
            gender = files[0]['sex']
            gender_counts[gender] += 1

    # 연령대별 통계
    age_group_counts = defaultdict(int)
    for files in person_data.values():
        if files:
            age_group = files[0]['age_group']
            age_group_counts[age_group] += 1

    # 통계 출력
    print("\n=== Person 데이터 분석 결과 ===")
    print(f"총 인원: {total_people}명")
    print(f"총 파일 수: {total_files}개")
    print(f"인당 최소 파일 수: {min_files}개")
    print(f"인당 최대 파일 수: {max_files}개")
    print(f"인당 평균 파일 수: {avg_files:.2f}개")

    print("\n=== 파일 수별 인원 분포 ===")
    for count in sorted(file_count_distribution.keys()):
        people_count = file_count_distribution[count]
        print(f"파일 {count}개: {people_count}명")

    print("\n=== 겹치는 파일 수가 많은 상위 10명 ===")
    top_people = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for person_id, count in top_people:
        print(f"Person ID: {person_id}, 파일 수: {count}개")

    print("\n=== 성별 분포 ===")
    for gender, count in gender_counts.items():
        print(f"{gender}: {count}명 ({count / total_people * 100:.2f}%)")

    print("\n=== 연령대별 분포 ===")
    for age_group, count in sorted(age_group_counts.items()):
        print(f"{age_group}대: {count}명 ({count / total_people * 100:.2f}%)")

    return {
        'total_people': total_people,
        'total_files': total_files,
        'file_count_distribution': dict(file_count_distribution),
        'top_people': top_people,
        'gender_counts': dict(gender_counts),
        'age_group_counts': dict(age_group_counts)
    }


def main():
    # TL 폴더 경로 설정 - 실제 경로로 수정 필요
    tl_folder = '/Users/chanyoungko/Desktop/245.개인 특정을 위한 자필과 모사 필기체 데이터/01-1.정식개방데이터/Training/02.라벨링데이터/TL'

    # 결과 출력 폴더
    output_folder = '/Users/chanyoungko/Desktop/필기체_분석_결과'

    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)

    # Person 데이터 수집
    print("JSON 파일에서 person 데이터 수집 중...")
    person_data = collect_person_data(tl_folder)

    # 결과 통계
    print(f"총 {len(person_data)}명의 ID를 찾았습니다.")

    # Person 데이터 분석
    stats = analyze_person_data(person_data)

    # 분석 결과 CSV 파일로 저장
    stats_csv = os.path.join(output_folder, 'person_stats.csv')
    with open(stats_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['통계 항목', '값'])
        writer.writerow(['총 인원', stats['total_people']])
        writer.writerow(['총 파일 수', stats['total_files']])
        writer.writerow(['파일 수별 인원 분포', ''])
        for count, people in sorted(stats['file_count_distribution'].items()):
            writer.writerow([f'파일 {count}개', f'{people}명'])

    print(f"통계 데이터 저장 완료: {stats_csv}")

    # CSV 파일로 내보내기
    output_csv = os.path.join(output_folder, 'person_id_mapping.csv')
    export_to_csv(person_data, output_csv)

    # Person ID별 폴더 생성
    print("Person ID별 폴더 생성 중...")
    create_person_folders(person_data, os.path.join(output_folder, 'person_folders'))

    print("작업 완료!")


if __name__ == "__main__":
    main()