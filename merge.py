import cv2
import os

# 이미지가 저장된 디렉토리와 결과 비디오 파일 경로
image_dir = '/home/elicer/parking_project/dataset/data/실외_대형주차장/val_src'
output_video_path = 'output_video.mp4'

# 이미지 찾기
image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
image_files.sort()  # 순서대로 정렬

# 첫 번째 이미지에 기반하여 비디오 초기화
first_image_path = os.path.join(image_dir, image_files[0])
first_image = cv2.imread(first_image_path)
height, width, layers = first_image.shape

# VideoWriter 객체 생성
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 또는 'XVID', 'X264' 등 코덱 사용
fps = 5  # 초당 프레임 수
video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# 이미지 추가
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    image = cv2.imread(image_path)
    video.write(image)  # 프레임 추가

# 비디오 파일 저장 종료
video.release()
print(f"Video saved as {output_video_path}")
