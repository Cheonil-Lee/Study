
# local

# 환경설정

생성 확인 

conda env list

가상환경 생성

conda create -n **yoloNsam2**  python==3.10

환경 활성화

conda activate **yoloNsam2**

---

새 가상환경에 ipykernel 다운로드

pip install ipykernel

커널추가

pip install jupyter notebook

python -m ipykernel install —user —name **yoloNsam2**—display-name “**yoloNsam2**”

**python: 3.10**

`pip install roboflow`

`conda install cudatoolkit`

`pip install ultralytics`

**pytorch 다운로드**

[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

![image](https://github.com/user-attachments/assets/be1d0763-031a-421a-8f70-34317e8a1ee2)


`conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`

**git 설치**

`conda install git pip`

## YOLOv8 GPU 구성

아나콘다 창에 **nvidia-smi** 입력하여 버전 확인

<img width="562" alt="image 1" src="https://github.com/user-attachments/assets/c53e6723-0ce4-4dd1-8dbb-73ba8135e53b" />


**anaconda prompt 경로 이동 후 sam2 설치**

- **sam2는 내가 작업할 디렉토리보다 상위에 있어야 함**
- **YOLOv8nSAM2** 폴더에 sam2 설치

<img width="263" alt="image 2" src="https://github.com/user-attachments/assets/d5f81208-7ec9-4823-9896-07ce6b3a19ab" />


sam2 설치 

`git clone [https://github.com/facebookresearch/sam2.git](https://github.com/facebookresearch/sam2.git) && cd sam2`

`pip install -e .`

jupyter, matplotlib 필요

`pip install -e ".[notebooks]”`

https://github.com/facebookresearch/sam2/blob/main/README.md 

위의 링크로 가서 

![image 3](https://github.com/user-attachments/assets/fd9de938-670c-473e-a2d8-4df7a17ad2d4)


sam2.1_hiera_large.pt 다운로드

다운로드된 sam2.1_hiera_large.pt를 sam2/**checkpoints** 폴더에 붙여넣기

![image 4](https://github.com/user-attachments/assets/3dcde1e8-3cf9-4156-b826-8e1f02a9f314)


작업하고 있는 ipynb를 sam2 폴더에 넣기

---

## CUDA설치, GPU 사용 가능 확인

```python
import torch
torch.cuda.is_available()
```

True

# YOLOv8

## **Roboflow Custom Dataset 로컬에 다운로드**

```python
from roboflow import Roboflow
rf = Roboflow(api_key="**********")
project = rf.workspace("humantest").project("yolov8_blink02")
version = project.version(1)
dataset = version.download("yolov8")               
```

![image 5](https://github.com/user-attachments/assets/c2572787-f2bf-487e-b9e8-7a740036308e)


**다운로드된 Custom Dataset 위치**

![image 6](https://github.com/user-attachments/assets/112c2b57-7617-4dfa-96b8-753ce99453a2)


## **주어진 경로의 data.yaml 파일 내용 확인**

```python
# 파일 경로 설정, 상대 경로 복사
file_path = 'yolov8_Blink02-1\data.yaml'

# 파일 내용 읽기 및 출력
with open(file_path, 'r') as file:
    content = file.read()
    print(content)
```

![image 7](https://github.com/user-attachments/assets/ae4525a9-d145-4272-ad12-df60a492cfbb)


## **커스텀 데이터에 맞는 YAML 파일 만들기**

```python
import yaml

data = { 'train' : 'yolov8_Blink02-1/train/images',
        'val' : 'yolov8_Blink02-1/valid/images', # Make sure this path is correct
         'test' : 'yolov8_Blink02-1/test/images',
         'names' : ['Blink', 'CAR', 'Truck'],
         'nc': 3}

with open('yolov8_Blink02-1/data.yaml', 'w') as f:
    yaml.dump(data, f)
```

## Install YOLOv8

```python
import ultralytics

ultralytics.checks()
```

Ultralytics 8.3.49  Python-3.10.16 torch-2.5.1 CUDA:0 (NVIDIA GeForce RTX 4050 Laptop GPU, 6140MiB)
Setup complete  (20 CPUs, 15.6 GB RAM, 382.7/476.0 GB disk)

## **Load a pre-trained model**

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
```

```python
print(type(model.names),len(model.names))

print(model.names)
```

<class 'dict'> 80
{0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', …

## YOLOv8 커스텀 데이터 학습하기

```python
model.train(data='yolov8_Blink02-1\data.yaml',epochs=100, patience = 32, imgsz=416),
```

![image 8](https://github.com/user-attachments/assets/7275dd2a-54b7-4768-8eab-2f4613b730b0)


# SAM2

```python
#필요한 패키지를 설치해주겠습니다
import torch
from PIL import Image
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
```

```python
import numpy as np
#gpu관련 설정해주겠습니다
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
#시각화를 위한 함수들 입니다.
def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
        # print(len(contours[0]))
        # # print('here')
        # # Img = Image.fromarray(((contours[0])* 255).astype(np.uint8),mode='RGBA')
        # # Img.save('./output.png',format="PNG")
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.savefig(f"Mask {i+1}")
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()
```

## 모델 설정

```python
sam2_checkpoint = "C:\\Users\\YOON\\Study\\YOLOv8nSAM2\\sam2\\checkpoints\\sam2.1_hiera_large.pt"
model_cfg = "C:\\Users\\YOON\\Study\\YOLOv8nSAM2\\sam2\\sam2\\configs\\sam2.1\\sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")

predictor = SAM2ImagePredictor(sam2_model)
```

## 결과 보기

```python
# YOLOv8 모델 로드
model = YOLO('C:\\Users\\YOON\\Study\\YOLOv8nSAM2\\sam2\\runs\\detect\\train\\weights\\best.pt')  # 학습된 모델 경로
```

```python
new_image_path = 'C:\\Users\\YOON\\Study\\YOLOv8nSAM2\\194556.png’
```

```python
# 이미지에서 사람 탐지
results = model.predict(source=new_image_path, save=True)

# 결과 시각화
for result in results:
    result.plot()  # 탐지된 결과를 시각화합니다.
```

![image 9](https://github.com/user-attachments/assets/23ecdf95-cf6d-4cb2-8f12-4ebe6b7aafa1)


**맨 밑 줄이 결과 이미지가 저장된 경로**

![image 10](https://github.com/user-attachments/assets/749cdbd7-f063-483b-b705-e6ec7343aa92)


**결과 이미지(194556.jpg)**

![194556](https://github.com/user-attachments/assets/85cc50fc-cdd1-42e1-9518-98b0bb473134)


```python
import numpy as np
from PIL import Image

# 새로운 이미지 경로 설정
image1 = new_image_path  

# 이미지 로드
image = Image.open(image1).convert("RGB")  # 이미지를 RGB 형식으로 변환
image_np = np.array(image)  # NumPy 배열로 변환

# HWC 형식 확인 및 img_batch 설정
if image_np.ndim == 3 and image_np.shape[2] == 3:  # HWC 확인
    img_batch = [image_np]  # NumPy 배열을 포함한 배치 생성
else:
    raise ValueError("이미지 형식이 올바르지 않습니다.")

# predictor에 이미지 배치 설정
predictor.set_image_batch(img_batch)
```

```python
# 점 좌표를 저장할 리스트
points = []
for result in results:
    for box in result.boxes.xyxy:  # 각 예측된 박스
        x_center = (box[0] + box[2]) / 2  # x 좌표의 중앙
        y_center = (box[1] + box[3]) / 2  # y 좌표의 중앙
        points.append((x_center.item(), y_center.item()))  # 점 추가
```

```python
%matplotlib inline
# 포인트들을 배치로 준비
pts_batch = [np.array(points)]  # YOLOv8에서 추출한 모든 점들을 포함
labels_batch = [np.array([1] * len(points))]  # 각 점에 대해 동일한 레이블 (예: 1)

# masks_batch와 scores_batch 예측
masks_batch, scores_batch, _ = predictor.predict_batch(pts_batch, labels_batch, multimask_output=True)

# 각 객체에 대해 가장 좋은 마스크 선택
best_masks = []
for masks, scores in zip(masks_batch, scores_batch):
    if len(masks) == len(scores):  # 길이 확인
        best_mask_index = np.argmax(scores, axis=-1)
        best_masks.append(masks[best_mask_index])  # 최고 점수의 마스크 선택
    else:
        print("Length mismatch: masks =", len(masks), "scores =", len(scores))

# 단일 이미지에 대한 마스크와 포인트 시각화
plt.figure(figsize=(10, 10))
plt.imshow(image)  # 원본 이미지 사용
"""
# 포인트 시각화
for point in points:
    plt.scatter(point[0], point[1], c='red', s=100, label='Detected Point')  # 포인트 시각화
"""

# 마스크 시각화
for mask in best_masks:
    if mask.ndim > 2:  # 차원 확인
        mask = mask.squeeze(0)  # 첫 번째 차원 제거
    plt.imshow(mask, alpha=0.5)  # 마스크 시각화 (반투명)
```

![image 11](https://github.com/user-attachments/assets/4b10c359-3665-45bc-8350-9a144db5312f)


## 원본 이미지 + mask

```python
# 원본 이미지 로드
image = Image.open(new_image_path).convert("RGB")  # 이미지를 RGB 형식으로 변환
image_np = np.array(image)  # NumPy 배열로 변환

# 포인트들을 배치로 준비
pts_batch = [np.array(points)]  # YOLOv8에서 추출한 모든 점들을 포함
labels_batch = [np.array([1] * len(points))]  # 각 점에 대해 동일한 레이블 (예: 1)

# masks_batch와 scores_batch 예측
masks_batch, scores_batch, _ = predictor.predict_batch(pts_batch, labels_batch, multimask_output=True)

# 각 객체에 대해 가장 좋은 마스크 선택
best_masks = []
for masks, scores in zip(masks_batch, scores_batch):
    if len(masks) == len(scores):  # 길이 확인
        best_mask_index = np.argmax(scores, axis=-1)
        best_masks.append(masks[best_mask_index])  # 최고 점수의 마스크 선택
    else:
        print("Length mismatch: masks =", len(masks), "scores =", len(scores))

# 단일 이미지에 대한 마스크 시각화
plt.figure(figsize=(10, 10))
plt.imshow(image_np)  # 원본 이미지 사용

# 마스크 시각화
for mask in best_masks:
    if mask.ndim > 2:  # 차원 확인
        mask = mask.squeeze(0)  # 첫 번째 차원 제거
    
    # 마스크를 원본 이미지와 동일한 크기로 변환
    mask_resized = np.resize(mask, image_np.shape[:2])  # HxW 형식으로 변환
    mask_resized = mask_resized.astype(np.uint8)  # uint8 형식으로 변환

    # 마스크를 컬러로 변환 (예: 빨간색)
    colored_mask = np.zeros_like(image_np)  # 원본 이미지와 같은 크기
    colored_mask[mask_resized > 0] = [255, 0, 0]  # 마스크가 있는 부분을 빨간색으로 설정

    # 원본 이미지와 컬러 마스크를 합성
    combined_image = np.where(colored_mask > 0, colored_mask, image_np)

    # 합성된 이미지를 시각화
    plt.imshow(combined_image, alpha=0.5)  # 반투명하게 표시

plt.axis('off')
plt.show()
```

![image 12](https://github.com/user-attachments/assets/31ade534-f8a5-4bf6-8f30-ead95af7db02)
