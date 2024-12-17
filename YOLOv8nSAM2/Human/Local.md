# human

# 환경 설정

→ https://github.com/Cheonil-Lee/Study/blob/main/YOLOv8nSAM2/CAR/Local.md

위와 동일 

---

## CUDA 설치, GPU 사용 가능 확인

```python
import torch
torch.cuda.is_available()
```

True

# YOLOv8

**Roboflow Custom Dataset 로컬에 다운로드**

```python
from roboflow import Roboflow
rf = Roboflow(api_key="**********")
project = rf.workspace("humantest").project("human1-wyh9n")
version = project.version(3)
dataset = version.download("yolov8")               
```

loading Roboflow workspace...
loading Roboflow project...

 **주어진 경로의 data.yaml 파일 내용 확인**

```python
# 파일 경로 설정, 상대 경로 복사
file_path = 'C:\\Users\\YOON\\Study\\YnS_human\\sam2\\human1-3\\data.yaml'

# 파일 내용 읽기 및 출력
with open(file_path, 'r') as file:
    content = file.read()
    print(content)
```

![image](https://github.com/user-attachments/assets/9b3b6283-45af-4175-a57e-968f2397e3a9)


**커스텀 데이터에 맞는 YAML 파일 만들기**

```python
import yaml

data = { 'train' : 'C:\\Users\\YOON\\Study\\YnS_human\\sam2\\human1-3\\train\\images',
        'val' : 'C:\\Users\\YOON\\Study\\YnS_human\\sam2\\human1-3\\valid\\images', # Make sure this path is correct
         'test' : 'C:\\Users\\YOON\\Study\\YnS_human\\sam2\\human1-3\\test\\images',
         'names' : ['human'],
         'nc': 1}

with open('C:\\Users\\YOON\\Study\\YnS_human\\sam2\\human1-3\\data.yaml', 'w') as f:
    yaml.dump(data, f)
```

**Install YOLOv8**

```python
import ultralytics

ultralytics.checks()
```

![image 1](https://github.com/user-attachments/assets/15dc121c-fa50-4e39-a9cc-66af01c0e02d)


**##### Load a pre-trained model**

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
```

![image 2](https://github.com/user-attachments/assets/5108bf64-1e1e-4a47-98bf-15d6e1332f10)


```python
print(type(model.names),len(model.names))

print(model.names)
```

![image 3](https://github.com/user-attachments/assets/c3711442-e55b-4ddb-849e-f0e11bfdfa9d)


**#####  YOLOv8 커스텀 데이터 학습하기**

```python
model.train(data='C:\\Users\\YOON\\Study\\YnS_human\\sam2\\human1-3\\data.yaml',epochs=100, patience = 32, imgsz=416),
```

![image 4](https://github.com/user-attachments/assets/20c1d904-7631-438a-9ac8-ca5cb70b8c42)


- 올바른 경로를 썼는지 확인
- 전체적인 디렉토리 점검 필요

![image 5](https://github.com/user-attachments/assets/359976e2-b080-4a7e-a006-d81ae9fa9014)


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

모델 설정

```python
sam2_checkpoint = "C:\\Users\\YOON\\Study\\YnS_human\\sam2\\checkpoints\\sam2.1_hiera_large.pt"
model_cfg = "C:\\Users\\YOON\\Study\\YnS_human\\sam2\\sam2\\configs\\sam2.1\\sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")

predictor = SAM2ImagePredictor(sam2_model)
```

결과 보기

```python
# 2. YOLOv8 모델 로드
model = YOLO('C:\\Users\\YOON\\Study\\YnS_human\\sam2\\runs\\detect\\train\\weights\\best.pt')  # 학습된 모델 경로
```

```python
new_image_path = 'C:\\Users\\YOON\\Study\\YnS_human\\20210625.jpg’
```

```python
# 이미지에서 사람 탐지
results = model.predict(source=new_image_path, save=True)

# 결과 시각화
for result in results:
    result.plot()  # 탐지된 결과를 시각화합니다.
```

![image 6](https://github.com/user-attachments/assets/89b5dcfa-d4a0-41e2-8117-eb633bbb9f13)


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
# 4. 점 좌표를 저장할 리스트
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

'''
image1_pts = np.array([[565, 260]])  # 첫 번째 포인트
pts_batch = [image1_pts]  # 하나의 포인트만 포함
labels_batch = [np.array([1])]  # 포인트에 대한 레이블
'''

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

# 포인트 시각화
for point in points:
    plt.scatter(point[0], point[1], c='red', s=100, label='Detected Point')  # 포인트 시각화

# 마스크 시각화
for mask in best_masks:
    if mask.ndim > 2:  # 차원 확인
        mask = mask.squeeze(0)  # 첫 번째 차원 제거
    plt.imshow(mask, alpha=0.5)  # 마스크 시각화 (반투명)

```

![image 7](https://github.com/user-attachments/assets/8775ce2d-f6b2-40a8-912b-bd0568063fae)


원본 이미지 + mask

```python
# 원본 이미지 로드
image = Image.open(new_image_path).convert("RGB")  # 이미지를 RGB 형식으로 변환
image_np = np.array(image)  # NumPy 배열로 변환

# 포인트들을 배치로 준비
pts_batch = [np.array(points)]  # YOLOv8에서 추출한 모든 점들을 포함
labels_batch = [np.array([1] * len(points))]  # 각 점에 대해 동일한 레이블 (예: 1)

'''
image1_pts = np.array([[565, 260]])  # 첫 번째 포인트
pts_batch = [image1_pts]  # 하나의 포인트만 포함
labels_batch = [np.array([1])]  # 포인트에 대한 레이블
'''

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

![image 8](https://github.com/user-attachments/assets/feb53f50-453c-40cc-a03b-6080f94d974f)


최종

```python
# 2. YOLOv8 모델 로드
model = YOLO('C:\\Users\\YOON\\Study\\YnS_human\\sam2\\runs\\detect\\train\\weights\\best.pt')  # 학습된 모델 경로

new_image_path = 'C:\\Users\\YOON\\Study\\YnS_human\\20210625.jpg'

# 이미지에서 사람 탐지
results = model.predict(source=new_image_path, save=True)

# 결과 시각화
for result in results:
    result.plot()  # 탐지된 결과를 시각화합니다.
```

![image 9](https://github.com/user-attachments/assets/dd45f454-38be-4be5-b30c-2a3a39089f7f)


```python
# 3. 새로운 이미지 로드
image = Image.open(new_image_path).convert("RGB")  # 이미지를 RGB 형식으로 변환
image_np = np.array(image)  # NumPy 배열로 변환

# HWC 형식 확인 및 img_batch 설정
if image_np.ndim == 3 and image_np.shape[2] == 3:  # HWC 확인
    img_batch = [image_np]  # NumPy 배열을 포함한 배치 생성
else:
    raise ValueError("이미지 형식이 올바르지 않습니다.")

# predictor에 이미지 배치 설정
predictor.set_image_batch(img_batch)

# 4. 점 좌표를 저장할 리스트
points = []
for result in results:
    for box in result.boxes.xyxy:  # 각 예측된 박스
        x_center = (box[0] + box[2]) / 2  # x 좌표의 중앙
        y_center = (box[1] + box[3]) / 2  # y 좌표의 중앙
        points.append((x_center.item(), y_center.item()))  # 점 추가

# 5. 각 이미지에 대한 포인트 정의
image_pts_list = [np.array([point]) for point in points]  # 바운딩 박스 중앙값을 포인트로 사용

# 6. 마스크 결합을 위한 초기화
combined_mask = None

# 각 이미지에 대해 마스크 예측 및 결합
for img_pts in image_pts_list:
    pts_batch = [img_pts]  # 포인트 배치
    labels_batch = [np.array([1])]  # 포인트에 대한 레이블

    masks_batch, scores_batch, _ = predictor.predict_batch(pts_batch, labels_batch, multimask_output=True)

    # 각 객체에 대해 가장 좋은 마스크 선택
    if masks_batch:  # masks_batch가 비어있지 않은 경우
        for masks, scores in zip(masks_batch, scores_batch):
            if len(masks) == len(scores):  # 길이 확인
                best_mask_index = np.argmax(scores, axis=-1)
                best_mask = masks[best_mask_index]  # 최고 점수의 마스크 선택

                # 마스크 결합
                if combined_mask is None:
                    combined_mask = best_mask
                else:
                    combined_mask = np.maximum(combined_mask, best_mask)  # 마스크 결합 (OR 연산)

# 단일 이미지에 대한 마스크와 포인트 시각화
plt.figure(figsize=(10, 10))
plt.imshow(image_np)  # 원본 이미지

# 포인트 시각화
for point in points:
    plt.scatter(point[0], point[1], c='red', s=100)  # 포인트 시각화

# 결합된 마스크 시각화
if combined_mask is not None:
    if combined_mask.ndim > 2:  # 차원 확인
        combined_mask = combined_mask.squeeze(0)  # 첫 번째 차원 제거
    show_mask(combined_mask, plt.gca(), random_color=True)  # 결합된 마스크 시각화

plt.axis('off')  # 축 숨기기
plt.show()  # 결과 표시
```

![image 10](https://github.com/user-attachments/assets/b2bb32e9-a35d-4ecb-a5df-9951908e7ac8)


포인트 시각화와 mask 비교

```python
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 나눔글꼴 경로 설정
font_path = 'C:/Windows/Fonts/HanSantteutDotum-Regular.ttf'

# 폰트 이름 가져오기
font_name = fm.FontProperties(fname=font_path).get_name()

# 폰트 설정
plt.rc('font', family=font_name)
```

```python
# 3. 새로운 이미지 로드
image = Image.open(new_image_path).convert("RGB")  # 이미지를 RGB 형식으로 변환
image_np = np.array(image)  # NumPy 배열로 변환

# HWC 형식 확인 및 img_batch 설정
if image_np.ndim == 3 and image_np.shape[2] == 3:  # HWC 확인
    img_batch = [image_np]  # NumPy 배열을 포함한 배치 생성
else:
    raise ValueError("이미지 형식이 올바르지 않습니다.")

# predictor에 이미지 배치 설정
predictor.set_image_batch(img_batch)

# 4. 점 좌표를 저장할 리스트
points = []
for result in results:
    for box in result.boxes.xyxy:  # 각 예측된 박스
        x_center = (box[0] + box[2]) / 2  # x 좌표의 중앙
        y_center = (box[1] + box[3]) / 2  # y 좌표의 중앙
        points.append((x_center.item(), y_center.item()))  # 점 추가

# 5. 각 이미지에 대한 포인트 정의
image_pts_list = [np.array([point]) for point in points]  # 바운딩 박스 중앙값을 포인트로 사용

# 6. 마스크 결합을 위한 초기화
combined_mask = None

# 각 이미지에 대해 마스크 예측 및 결합
for img_pts in image_pts_list:
    pts_batch = [img_pts]  # 포인트 배치
    labels_batch = [np.array([1])]  # 포인트에 대한 레이블

    masks_batch, scores_batch, _ = predictor.predict_batch(pts_batch, labels_batch, multimask_output=True)

    # 각 객체에 대해 가장 좋은 마스크 선택
    if masks_batch:  # masks_batch가 비어있지 않은 경우
        for masks, scores in zip(masks_batch, scores_batch):
            if len(masks) == len(scores):  # 길이 확인
                best_mask_index = np.argmax(scores, axis=-1)
                best_mask = masks[best_mask_index]  # 최고 점수의 마스크 선택

                # 마스크 결합
                if combined_mask is None:
                    combined_mask = best_mask
                else:
                    combined_mask = np.maximum(combined_mask, best_mask)  # 마스크 결합 (OR 연산)

# 첫 번째 이미지에 대한 포인트와 마스크 시각화
plt.figure(figsize=(10, 10))
plt.imshow(image_np)  # 원본 이미지

# 포인트 시각화
for point in points:
    plt.scatter(point[0], point[1], c='red', s=100)  # 포인트 시각화

# 결합된 마스크 시각화
if combined_mask is not None:
    if combined_mask.ndim > 2:  # 차원 확인
        combined_mask = combined_mask.squeeze(0)  # 첫 번째 차원 제거
    show_mask(combined_mask, plt.gca(), random_color=True)  # 결합된 마스크 시각화

plt.axis('off')  # 축 숨기기
plt.title("포인트 시각화")
plt.show()  # 결과 표시

# 두 번째 이미지 (포인트 시각화 없이)
plt.figure(figsize=(10, 10))
plt.imshow(image_np)  # 원본 이미지

# 결합된 마스크 시각화
if combined_mask is not None:
    if combined_mask.ndim > 2:  # 차원 확인
        combined_mask = combined_mask.squeeze(0)  # 첫 번째 차원 제거
    show_mask(combined_mask, plt.gca(), random_color=True)  # 결합된 마스크 시각화

plt.axis('off')  # 축 숨기기
plt.title("포인트 숨김")
plt.show()  # 결과 표시
```

![image 11](https://github.com/user-attachments/assets/7e44ea91-d37d-4736-8287-7f66786696a5)

![image 12](https://github.com/user-attachments/assets/7c8d34b4-5a9a-42dc-ab6b-b62b872993bf)
