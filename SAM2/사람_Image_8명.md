# 사람 Image_8명

```python
import torch
torch.cuda.is_available()
```

```python
#필요한 패키지를 설치해주겠습니다
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
```

```python
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

## 이미지 받아오기

```python
image = Image.open('20210625.jpg')
image = np.array(image.convert("RGB"))
```

## 모델 설정

```python
sam2_checkpoint = "C:\\Users\\YOON\\Study\\SAM2\\checkpoints\\sam2.1_hiera_large.pt"
model_cfg = "C:\\Users\\YOON\\Study\sam2\\sam2\\configs\\sam2.1\\sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")

predictor = SAM2ImagePredictor(sam2_model)
```

## 이미지 확인

```python
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('on')
plt.show()
```

![image](https://github.com/user-attachments/assets/cc760772-14ae-461d-92d8-7992340452c6)


```python
predictor.set_image(image)
```

```python
input_point = np.array([[565, 260]])
input_label = np.array([1])
```

```python
plt.figure(figsize=(10, 10))
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.show()
```

![image 1](https://github.com/user-attachments/assets/96de4b5e-c4c0-4eb1-b557-88a3f597e2af)


```python
print(predictor._features["image_embed"].shape, predictor._features["image_embed"][-1].shape)
```

torch.Size([1, 256, 64, 64]) torch.Size([256, 64, 64])

```python
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)
sorted_ind = np.argsort(scores)[::-1]
masks = masks[sorted_ind]
scores = scores[sorted_ind]
logits = logits[sorted_ind]
```

```python
masks.shape  # (number_of_masks) x H x W
```

(3, 518, 700)

```python
show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)
```

![image 2](https://github.com/user-attachments/assets/04ec51a5-1d4f-4bfc-bf2c-e4c092f8d92b)

![image 3](https://github.com/user-attachments/assets/a66839f2-d084-4efa-b15c-2ad7c31aec3c)

![image 4](https://github.com/user-attachments/assets/5d08a60d-ffbe-4445-ac07-901c124bad76)


## point로 mask 찾기

```python
image1 = image  # truck.jpg from above
img_batch = [image1]
```

```python
predictor.set_image_batch(img_batch)
```

```python
image1 = image  
image1_pts = np.array([
    [[565, 260]],
    [[555, 150]]
    ]) 
image1_labels = np.array([[1], [1]])

pts_batch = [image1_pts]
labels_batch = [image1_labels]
```

```python
masks_batch, scores_batch, _ = predictor.predict_batch(pts_batch, labels_batch, multimask_output=True)

# Select the best single mask per object
best_masks = []
for masks, scores in zip(masks_batch,scores_batch):
    best_masks.append(masks[range(len(masks)), np.argmax(scores, axis=-1)])
```

```python
for image, points, labels, masks in zip(img_batch, pts_batch, labels_batch, best_masks):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask, plt.gca(), random_color=True)
    show_points(points, labels, plt.gca())
```

![image 5](https://github.com/user-attachments/assets/53f35d8a-fff6-480a-9b4e-a481be921bfb)


```python
# 각 포인트 정의 (8개 포인트)
image1_pts = np.array([[565, 260]])  # 첫 번째 포인트
image2_pts = np.array([[450, 220]])  # 두 번째 포인트
image3_pts = np.array([[395, 250]])  # 세 번째 포인트
image4_pts = np.array([[345, 300]])  # 네 번째 포인트
image5_pts = np.array([[260, 310]])  # 다섯 번째 포인트
image6_pts = np.array([[210, 400]])  # 여섯 번째 포인트
image7_pts = np.array([[50, 300]])  # 일곱 번째 포인트
image8_pts = np.array([[50, 400]])  # 여덟 번째 포인트
```

```python
# 이미지와 포인트 정의
image1 = image  
image2 = image  
image3 = image  
image4 = image  
image5 = image  
image6 = image  
image7 = image  
image8 = image  

# 각 이미지에 대한 포인트 정의
image1_pts = np.array([[565, 260]])  # 첫 번째 포인트
image2_pts = np.array([[450, 220]])  # 두 번째 포인트
image3_pts = np.array([[395, 250]])  # 세 번째 포인트
image4_pts = np.array([[345, 300]])  # 네 번째 포인트
image5_pts = np.array([[260, 310]])  # 다섯 번째 포인트
image6_pts = np.array([[210, 400]])  # 여섯 번째 포인트
image7_pts = np.array([[50, 300]])   # 일곱 번째 포인트
image8_pts = np.array([[50, 400]])   # 여덟 번째 포인트

# 모든 포인트를 리스트에 저장
pts_list = [
    image1_pts,
    image2_pts,
    image3_pts,
    image4_pts,
    image5_pts,
    image6_pts,
    image7_pts,
    image8_pts
]

# 모든 이미지를 리스트에 저장
images = [
    image1,
    image2,
    image3,
    image4,
    image5,
    image6,
    image7,
    image8
]

# 마스크 결합을 위한 초기화
combined_mask = None

# 각 이미지에 대해 마스크 예측 및 결합
for img, pts in zip(images, pts_list):
    pts_batch = [pts]  # 하나의 포인트만 포함
    labels_batch = [np.array([1])]  # 포인트에 대한 레이블

    masks_batch, scores_batch, _ = predictor.predict_batch(pts_batch, labels_batch, multimask_output=True)

    # 각 객체에 대해 가장 좋은 마스크 선택
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
plt.imshow(image1)  # 원본 이미지
"""
# 포인트 시각화
for pts in pts_list:
    for point in pts:
        plt.scatter(point[0], point[1], c='red', s=100)  # 포인트 시각화
"""
# 결합된 마스크 시각화
if combined_mask is not None:
    if combined_mask.ndim > 2:  # 차원 확인
        combined_mask = combined_mask.squeeze(0)  # 첫 번째 차원 제거
    show_mask(combined_mask, plt.gca(), random_color=True)  # 결합된 마스크 시각화

plt.axis('off')  # 축 숨기기
#plt.title('Combined Masks for All Points with Point Locations')
plt.show()  # 결과 표시

```

![image 6](https://github.com/user-attachments/assets/d47c4e22-e490-4946-ab87-39f76f8922bf)


```python
# 이미지와 포인트, 레이블 정의
image1 = image  

# 포인트들을 배치로 준비
pts_batch = [image1_pts]  # 하나의 포인트만 포함
labels_batch = [np.array([1])]  # 포인트에 대한 레이블

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
plt.imshow(image1)  # image1 사용

# 포인트 시각화
for point, label in zip(pts_batch, labels_batch):
    for p in point:
        plt.scatter(p[0], p[1], c='red', s=100, label=f'Label: {label[0]}')  # 포인트 시각화

# 마스크 시각화
for mask in best_masks:
    if mask.ndim > 2:  # 차원 확인
        mask = mask.squeeze(0)  # 첫 번째 차원 제거
    show_mask(mask, plt.gca(), random_color=True)  # 마스크 시각화

plt.axis('off')  # 축 숨기기
plt.show()  # 결과 표시

```

![image 7](https://github.com/user-attachments/assets/98c5aede-3fd7-48a1-84aa-12543c80f436)


```python
# 포인트들을 배치로 준비
#image7_pts, 숫자 번호가 사람 번호
pts_batch = [image1_pts]  # 하나의 포인트만 포함
labels_batch = [np.array([1])]  # 포인트에 대한 레이블

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
plt.imshow(image1)  # image1 사용
"""
# 포인트 시각화
for point, label in zip(pts_batch, labels_batch):
    for p in point:
        plt.scatter(p[0], p[1], c='red', s=100, label=f'Label: {label[0]}')  # 포인트 시각화
"""

# 마스크 시각화
for mask in best_masks:
    if mask.ndim > 2:  # 차원 확인
        mask = mask.squeeze(0)  # 첫 번째 차원 제거
    show_mask(mask, plt.gca(), random_color=True)  # 마스크 시각화

plt.axis('off')  # 축 숨기기
plt.show()  # 결과 표시

```

![image 8](https://github.com/user-attachments/assets/c668cae8-0a42-4dcd-9109-49883691dc89)


# 결과

### 사람1

![image 9](https://github.com/user-attachments/assets/ee85b42e-1ebe-4ca0-860b-a96fd5310ad3)


### 사람2

![image 10](https://github.com/user-attachments/assets/a43c27d0-dbac-4ed7-9097-7d811bdb3ccc)


### 사람3

![image 11](https://github.com/user-attachments/assets/b4301b0c-79f3-4f18-bee5-ee07a325b7a7)


### 사람4

![image 12](https://github.com/user-attachments/assets/fe7d381b-0504-409a-ae00-939652406433)


### 사람5

![image 13](https://github.com/user-attachments/assets/5c663a4f-51da-4506-b95e-deb9bd49b4d5)


### 사람6

![image 14](https://github.com/user-attachments/assets/e34ba25f-c114-4014-9d8e-e66c12d126cf)


### 사람7

![image 15](https://github.com/user-attachments/assets/25fc7029-08b4-4873-bf84-2871d24f41c7)


### 사람8

![image 16](https://github.com/user-attachments/assets/ba8b77cd-2121-4f9a-bd18-496d900b0306)


### 전체

![image 17](https://github.com/user-attachments/assets/62559175-8b5e-4ad0-9974-2b6944f1a350)
