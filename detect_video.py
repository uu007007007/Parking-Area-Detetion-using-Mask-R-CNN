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
class_names =["a","a","a","a","a","a","a","a","a","a","a","a","a","a","Driveable Space", "a", "Parking Space"]

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
output_dir = './results'

# 디렉토리 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Mask R-CNN 모델 초기화 및 가중치 로드
inference_config = InferenceConfig()
test_model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir='')

test_model.load_weights(model_path, by_name=True)
print('Model load completed')


cap = cv2.VideoCapture('/home/elicer/parking_project/tyhsin/parking/output_video.mp4')

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), 
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):  # 채널에 대해 반복
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def display_results(image, boxes, masks, class_ids, class_names, scores):
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
        
        
        
        
while cap.isOpened():
    ret, img = cap.read()

    if not ret:
        break

    results = test_model.detect([img])

    rois = results[0]['rois']
    class_ids = results[0]['class_ids']
    scores = results[0]['scores']
    masks = results[0]['masks']

    result_img = img.copy()
    
    results = test_model.detect([result_img], verbose=1)
    r = results[0]

    # 결과 이미지 생성 및 저장
    display_results(result_img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

    out.write(result_img)

cap.release()
out.release()
