# Local

https://github.com/facebookresearch/sam2/blob/main/README.md

# Anaconda 환경설정

생성 확인 

conda env list

가상환경 생성

conda create -n **sam2Test**  python==3.10

환경 활성화

conda activate **sam2Test**

---

새 가상환경에 ipykernel 다운로드

pip install ipykernel

커널추가

pip install jupyter notebook

python -m ipykernel install —user —name **sam2Test**—display-name “**sam2Test**”

# local 환경설정

python >= 3.10

torch >= 2.5.1

PyTorch

`conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`

- 현재 Windows용 PyTorch는 Python 3.9-3.12만 지원합니다.

**cudatoolkit설치**

`conda install cudatoolkit`

**anaconda prompt 경로 이동 후 sam2 설치**

<img width="227" alt="image" src="https://github.com/user-attachments/assets/aa509221-f59f-4c18-afe4-49384b72964b" />


**(수정) sam2는 내가 작업할 디렉토리보다 상위에 있어야 함. 아래 참고**

![image 1](https://github.com/user-attachments/assets/9533aadf-69c2-4306-883e-09b5ddebb5c4)

sam2 설치 

`git clone [https://github.com/facebookresearch/sam2.git](https://github.com/facebookresearch/sam2.git) && cd sam2`

`pip install -e .`

**git 설치**

`conda install git pip`

jupyter, matplotlib 필요

`pip install -e ".[notebooks]”`

위의 git 링크로 가서 

![image 2](https://github.com/user-attachments/assets/246693d3-aadf-4d76-abbd-b692c6322c4b)


sam2.1_hiera_large.pt 다운로드

다운로드된 sam2.1_hiera_large.pt를 sam2/**checkpoints** 폴더에 붙여넣기

![image 3](https://github.com/user-attachments/assets/cec4b3f7-153a-4f60-81e9-f08b308eb1ed)


# 실행 코드

## CUDA설치, GPU 사용 가능 확인

```python
import torch
torch.cuda.is_available()
```

True

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
sam2_checkpoint = "C:\\Users\\YOON\\Study\\SAM2\\sam2\\checkpoints\\sam2.1_hiera_large.pt"
model_cfg = "C:\\Users\\YOON\\Study\\SAM2\\sam2\\sam2\\configs\\sam2.1\\sam2.1_hiera_l.yaml"

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

![image 4](https://github.com/user-attachments/assets/e9a490b1-b330-4a04-8823-bd0f689b226d)


```python
predictor.set_image(image)
```

```python
input_point = np.array([[395, 250]])
input_label = np.array([1])
```

```python
plt.figure(figsize=(10, 10))
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.show()
```

![image 5](https://github.com/user-attachments/assets/e7a29869-129e-4cb0-9e5e-d60f9d2277e4)


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

![image 6](https://github.com/user-attachments/assets/c3389c45-a23d-4422-982d-7dc173897a46)

![image 7](https://github.com/user-attachments/assets/7d5685c5-9b70-44cb-9e6f-9634fbbb0ff6)

![image 8](https://github.com/user-attachments/assets/159b2562-24f0-4b4b-a88c-6ec09fc386b1)

## **Specifying a specific object with additional points**

```python
import numpy as np

# 오른쪽 아이에서 왼쪽 아이로 (1, 2번째)
input_point = np.array([[565, 250], [450, 220]])
# 오른쪽 아이에서 왼쪽 아이로 (3, 4번째)
input_point2 = np.array([[395, 250], [345, 300]])

# n개의 포인트에 대한 레이블 설정
input_label = np.array([1, 1])  # input_point에 대한 레이블
input_label2 = np.array([1, 1])  # input_point2에 대한 레이블

# 모델의 가장 좋은 마스크 선택
mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask

# 첫 번째 포인트에 대한 예측 호출
masks1, scores1, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    mask_input=mask_input[None, :, :],
    multimask_output=False,
)

# 두 번째 포인트에 대한 예측 호출
masks2, scores2, _ = predictor.predict(
    point_coords=input_point2,
    point_labels=input_label2,
    mask_input=mask_input[None, :, :],
    multimask_output=False,
)

# input_point3를 생성하여 예측 호출
input_point3 = np.vstack((input_point, input_point2))
input_label3 = np.array([1, 1, 1, 1])  # input_point3에 대한 레이블

# input_point3에 대한 예측 호출
masks3, scores3, _ = predictor.predict(
    point_coords=input_point3,
    point_labels=input_label3,
    mask_input=mask_input[None, :, :],
    multimask_output=False,
)

# 각 마스크를 시각화 (각기 다른 그림으로)
# 첫 번째 마스크 시각화
show_masks(image, masks1, scores1, point_coords=input_point, input_labels=input_label)

# 두 번째 마스크 시각화
show_masks(image, masks2, scores2, point_coords=input_point2, input_labels=input_label2)

# 세 번째 마스크 시각화
show_masks(image, masks3, scores3, point_coords=input_point3, input_labels=input_label3)

```
![image 9](https://github.com/user-attachments/assets/bbd5a09c-72c8-46ce-b5dc-45af046a2cb7)

![image 10](https://github.com/user-attachments/assets/152055dd-3599-416f-93d5-8be45d21be94)

![image 11](https://github.com/user-attachments/assets/dfe2ba47-e193-410a-aace-e6bbd3e33a4f)


```python
import numpy as np

# 오른쪽 아이에서 왼쪽 아이로 (5, 6번째)
input_point_2 = np.array([[260, 310], [210, 400]])
# 오른쪽 아이에서 왼쪽 아이로 (7, 8번째)
input_point2_2 = np.array([[50, 300], [50, 400]])

# n개의 포인트에 대한 레이블 설정
input_label_2 = np.array([1, 1])  # input_point_2에 대한 레이블
input_label2_2 = np.array([1, 1])  # input_point2_2에 대한 레이블

# 모델의 가장 좋은 마스크 선택
mask_input_2 = logits[np.argmax(scores), :, :]  # Choose the model's best mask

# 첫 번째 포인트에 대한 예측 호출
masks1_2, scores1_2, _ = predictor.predict(
    point_coords=input_point_2,
    point_labels=input_label_2,
    mask_input=mask_input_2[None, :, :],
    multimask_output=False,
)

# 두 번째 포인트에 대한 예측 호출
masks2_2, scores2_2, _ = predictor.predict(
    point_coords=input_point2_2,
    point_labels=input_label2_2,
    mask_input=mask_input_2[None, :, :],
    multimask_output=False,
)

# input_point3_2를 생성하여 예측 호출
input_point3_2 = np.vstack((input_point_2, input_point2_2))
input_label3_2 = np.array([1, 1, 1, 1])  # input_point3_2에 대한 레이블

# input_point3_2에 대한 예측 호출
masks3_2, scores3_2, _ = predictor.predict(
    point_coords=input_point3_2,
    point_labels=input_label3_2,
    mask_input=mask_input_2[None, :, :],
    multimask_output=False,
)

# 각 마스크를 시각화 (각기 다른 그림으로)
# 첫 번째 마스크 시각화
show_masks(image, masks1_2, scores1_2, point_coords=input_point_2, input_labels=input_label_2)

# 두 번째 마스크 시각화
show_masks(image, masks2_2, scores2_2, point_coords=input_point2_2, input_labels=input_label2_2)

# 세 번째 마스크 시각화
show_masks(image, masks3_2, scores3_2, point_coords=input_point3_2, input_labels=input_label3_2)

```

![image 12](https://github.com/user-attachments/assets/62aabe37-99c9-4af0-8a6a-9ac49210fe65)

![image 13](https://github.com/user-attachments/assets/7e828c3e-3d4b-4b2f-8b49-7ccf2025d573)

![image 14](https://github.com/user-attachments/assets/e41c0a4d-7b5a-4f20-a092-fb243ed24d61)


```python
# 7번째 아이
input_point = np.array([[50, 300], [60, 340]])
#input_point2 = np.array([[395, 250], [345, 300]])

# n개의 포인트에 대한 레이블
input_label = np.array([1, 1])

mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
```

```python
masks, scores, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    mask_input=mask_input[None, :, :],
    multimask_output=False,
)
```

```python
masks.shape
```

(1, 518, 700)

show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label)

![image 15](https://github.com/user-attachments/assets/cf26fb44-61de-4b73-9c37-abc2a0c2d885)


```python
input_point = np.array([[565, 260], [565, 350]])
input_label = np.array([1, 0])

mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
```

```python
masks, scores, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    mask_input=mask_input[None, :, :],
    multimask_output=False,
)
```

```python
show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label)
```

![image 16](https://github.com/user-attachments/assets/e4157deb-0739-43ab-91d0-b189e537de26)


## **Specifying a specific object with a box**

```python
#input_box = np.array([310, 370, 380, 220])
#input_box = np.array([100, 600, 420, 100])
input_box = np.array([0, 530, 150, 270])
```

```python
masks, scores, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=False,
)
```

```python
show_masks(image, masks, scores, box_coords=input_box)
```

![image 17](https://github.com/user-attachments/assets/8430321a-cdd7-47a6-95c2-488f8763851c)


## **Combining points and boxes**

```python
input_box = np.array([440, 450, 630, 70])
input_point = np.array([[565, 260]])
input_label = np.array([0])
```

```python
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    box=input_box,
    multimask_output=False,
)
```

```python
show_masks(image, masks, scores, box_coords=input_box, point_coords=input_point, input_labels=input_label)
```

![image 18](https://github.com/user-attachments/assets/a76fb171-9382-44bf-99fd-ad33b03da3e8)


## **Batched prompt inputs**

```python
input_boxes = np.array([
    [490, 330, 610, 150],
    [500, 210, 590, 100],
    [450, 390, 600, 280],
    [450, 450, 550, 310],
])

#몸통
#얼굴
#바지
#신발
```

```python
masks, scores, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)
```

```python
masks.shape  # (batch_size) x (num_predicted_masks_per_input) x H x W
```

(4, 1, 518, 700)

```python
plt.figure(figsize=(10, 10))
plt.imshow(image)
for mask in masks:
    show_mask(mask.squeeze(0), plt.gca(), random_color=True)
for box in input_boxes:
    show_box(box, plt.gca())
plt.axis('off')
plt.show()
```

![image 19](https://github.com/user-attachments/assets/c57e0a0b-92a6-4e4d-a77b-b91dadda14a3)


## **End-to-end batched inferenc**

```python
import urllib.request
import os

# 다운로드할 URL
url = "https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/images/groceries.jpg"

# 파일을 저장할 경로
file_path = "groceries.jpg"

# 파일 다운로드
urllib.request.urlretrieve(url, file_path)

print("다운로드 완료:", file_path)

```

다운로드 완료: groceries.jpg

```python
image1 = image  # truck.jpg from above
image1_boxes = np.array([
    [490, 330, 610, 150],
    [500, 210, 590, 100],
    [450, 390, 600, 280],
    [450, 450, 550, 310],
])

image2 = Image.open('groceries.jpg')
image2 = np.array(image2.convert("RGB"))
image2_boxes = np.array([
    [450, 170, 520, 350],
    [350, 190, 450, 350],
    [500, 170, 580, 350],
    [580, 170, 640, 350],
])

img_batch = [image1, image2]
boxes_batch = [image1_boxes, image2_boxes]
```

```python
predictor.set_image_batch(img_batch)
```

```python
masks_batch, scores_batch, _ = predictor.predict_batch(
    None,
    None,
    box_batch=boxes_batch,
    multimask_output=False
)
```

```python
for image, boxes, masks in zip(img_batch, boxes_batch, masks_batch):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.squeeze(0), plt.gca(), random_color=True)
    for box in boxes:
        show_box(box, plt.gca())
```

![image 20](https://github.com/user-attachments/assets/6dd6a704-6f34-44ed-a038-7f5f711203d2)

![image 21](https://github.com/user-attachments/assets/cddaa65e-375e-4889-bef4-4902f75477ae)


## **point로 mask 찾기**

```python
image1 = image  # truck.jpg from above
image1_pts = np.array([
    [[565, 260]],
    [[565, 350]]
    ]) # Bx1x2 where B corresponds to number of objects
image1_labels = np.array([[1], [1]])

image2_pts = np.array([
    [[400, 300]],
    [[630, 300]],
])
image2_labels = np.array([[1], [1]])

pts_batch = [image1_pts, image2_pts]
labels_batch = [image1_labels, image2_labels]
```

```python
masks_batch, scores_batch, _ = predictor.predict_batch(pts_batch, labels_batch, box_batch=None, multimask_output=True)

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

![image 22](https://github.com/user-attachments/assets/71babf4d-1b45-41d5-97c8-b81adb717e8d)

![image 23](https://github.com/user-attachments/assets/e5671b5b-01ff-4012-8d87-554289069d89)
