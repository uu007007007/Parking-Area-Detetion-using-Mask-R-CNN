import os
import sys
import numpy as np
import skimage.io
import cv2
import time
ROOT_DIR = 'Mask_RCNN'  # Mask R-CNN 모듈의 루트 경로
sys.path.append(ROOT_DIR)
import mrcnn.model as modellib
from mrcnn.config import Config

# 절대 경로 지정

MODEL_DIR = os.path.join(ROOT_DIR, "logs")


# 클래스 이름 정의
class_names = ["a","a","a","a","a","a","a","a","a","a","a","a","a","a","Driveable Space", "a", "Parking Space"]

# Config 정의
class InferenceConfig(Config):
    # Give the configuration a recognizable name
    NAME = "custom"
    
    # Train on 1 GPU and 1 image per GPU. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    LEARNING_RATE = 0.001
    DETECTION_MIN_CONFIDENCE = 0.9
    # Number of classes (including background)
    NUM_CLASSES = 1 + 16  # background + 8 (Car,Van,Other Vehicle,Traffic Pole,Parking Block,Parking Sign,Driveable Space,Disabled Parking Space)

    # IMAGE_META_SIZE = 28

    # All of our training images are 1920x1012
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Matterport originally used resnet101, but I downsized to fit it on my graphics card
    BACKBONE = 'resnet50' # resnet50

    # To be honest, I haven't taken the time to figure out what these do
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 50
    POST_NMS_ROIS_INFERENCE = 500
    POST_NMS_ROIS_TRAINING = 1000

# 모델 경로와 이미지 디렉토리 정의
model_path = "/home/elicer/parking_project/tyhsin/parking/Mask_RCNN/logs/custom20240826T0716/mask_rcnn_custom_0004.h5"
real_test_dir = '/home/elicer/parking_project/dataset/data/실외_대형주차장/val_src'
output_dir = './results'

# 디렉토리 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Mask R-CNN 모델 초기화 및 가중치 로드
inference_config = InferenceConfig()
test_model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir='')

test_model.load_weights(model_path, by_name=True)
print('Model load completed')


# 이미지 경로 가져오기
image_paths = []
for filename in os.listdir(real_test_dir):
    if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
        image_paths.append(os.path.join(real_test_dir, filename))

# OpenCV를 사용하여 감지 결과를 이미지에 그리고 저장
def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):  # 채널에 대해 반복
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def display_results(image, boxes, masks, class_ids, class_names, scores, output_path):
    n_instances = boxes.shape[0]
    if not n_instances:
        print('No instances found!')
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    for i in range(n_instances):
        box = boxes[i]
        mask = masks[:, :, i]
        class_id = class_ids[i]
        score = scores[i]
        color = np.random.rand(3)

        # 마스크 적용
        image = apply_mask(image, mask, color)
        
        # 바운딩 박스 그리기
        y1, x1, y2, x2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), color * 255, 2)

        # 클래스 및 점수 표시
        label = f"{class_names[class_id]}: {score:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color * 255, 1)

    # 이미지 저장
    cv2.imwrite(output_path, image[..., ::-1])

# 결과를 생성하고 저장
for idx, image_path in enumerate(image_paths):  # 처리할 이미지의 수 지정
    start_time = time.time()
    image = skimage.io.imread(image_path)
    results = test_model.detect([image], verbose=1)
    r = results[0]
    end_time = time.time()
    take_time = end_time -start_time
    print(f'{take_time} 초 걸렸습니다.')

    # 결과 이미지 경로 설정
    output_path = os.path.join(output_dir, f"result_{idx}.jpg")

    # 결과 이미지 생성 및 저장
    display_results(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], output_path)

print("Results saved in", output_dir)
