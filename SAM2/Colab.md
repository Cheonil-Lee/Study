# Colab

https://github.com/facebookresearch/sam2?tab=readme-ov-file

![image](https://github.com/user-attachments/assets/593bbe76-0288-484d-ab3b-9abc2cff2d00)

sam2.1_hiera_large.pt 다운로드해서 드라이브에 넣기

sam2 다운로드 안 될 때 (오류 발생 시)

```python
!wget -P images [https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/images/groceries.jpg](https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/images/groceries.jpg)
```

---

## Google Drive 마운트

```python
# Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')
```

## sam2 다운로드

```python
using_colab = False
```

```python
if using_colab:
    import torch
    import torchvision
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA is available:", torch.cuda.is_available())
    import sys
    !{sys.executable} -m pip install opencv-python matplotlib

    !mkdir -p images
    #!wget -P images https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/images/truck.jpg
    !wget -P images https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/images/groceries.jpg

    !mkdir -p ../checkpoints/
    !wget -P ../checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

## set-up

```python
import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
```

```python
# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )
```

using device: cuda

```python
np.random.seed(3)

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
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()
```

## Example image

```python
# 필요한 라이브러리 임포트
from google.colab import files
import numpy as np
from PIL import Image
import io

# 이미지 파일 업로드
image_file = files.upload()

# 업로드한 이미지 파일을 읽기
image = io.BytesIO(image_file[list(image_file.keys())[0]])
image = np.array(Image.open(image))

# 결과 확인
print(image.shape)  # 이미지의 shape 출력
```

```python
image = Image.open('/content/20210625.jpg')
image = np.array(image.convert("RGB"))
```

```python
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('on')
plt.show()
```

![image 1](https://github.com/user-attachments/assets/d864ed68-b226-4ad3-8c1b-75968a399da3)


## **Selecting objects with SAM 2**

```python
# Colab에서 사용할 경우
import sys

# 필요한 패키지 설치
!{sys.executable} -m pip install opencv-python matplotlib
!{sys.executable} -m pip install git+https://github.com/facebookresearch/sam2.git

# 설치 확인
try:
    import sam2
    print("sam2 모듈이 성공적으로 설치되었습니다.")
except ModuleNotFoundError:
    print("sam2 모듈을 찾을 수 없습니다. 설치에 문제가 있을 수 있습니다.")

```

![image 2](https://github.com/user-attachments/assets/8a2c3ce5-8369-4a87-83b0-f6905ae0e326)


```python
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_checkpoint = "/content/drive/MyDrive/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)
```

```python
predictor.set_image(image)
```

## image에 point 생성

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

![image 3](https://github.com/user-attachments/assets/8e0f43e1-392a-4811-8052-5e281f20db6e)


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

![image 4](https://github.com/user-attachments/assets/177dac08-44ac-461c-bf85-cf6e7ed547de)

![image 5](https://github.com/user-attachments/assets/766dd748-f729-4aa2-8e00-237c96f22502)

![image 6](https://github.com/user-attachments/assets/1d5dd0ad-9bfe-4da6-bf59-4394761a2ffa)

## **Specifying a specific object with additional points**

```python
input_point = np.array([[565, 350], [565, 180]])
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

```python
show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label)
```

![image 7](https://github.com/user-attachments/assets/56628eac-6c65-4447-b186-72d7f6bb0e35)


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

![image 8](https://github.com/user-attachments/assets/52b7298e-d944-4eb6-b916-f3fa0409faa7)


## **Specifying a specific object with a box**

```python
input_box = np.array([490, 330, 610, 150])
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

![image 9](https://github.com/user-attachments/assets/df374625-346f-4567-bda1-a767cdc226af)


## **Combining points and boxes**

```python
input_box = np.array([490, 330, 610, 150])
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

![image 10](https://github.com/user-attachments/assets/407878f1-df1c-43a4-9897-8d3a32a03dbb)


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

![image 11](https://github.com/user-attachments/assets/98c3261c-5313-4d48-9a0a-12e7316c7651)


## **End-to-end batched inference**

```python
image1 = image  # truck.jpg from above
image1_boxes = np.array([
    [490, 330, 610, 150],
    [500, 210, 590, 100],
    [450, 390, 600, 280],
    [450, 450, 550, 310],
])

image2 = Image.open('/content/images/groceries.jpg')
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

## predictor.set_image_batch(img_batch)

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

![image 12](https://github.com/user-attachments/assets/5b878bc3-72b7-49c3-aee3-fe67b00cd84b)

![image 13](https://github.com/user-attachments/assets/5086c973-2305-4e78-a982-3cb960a29db3)


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

![image 14](https://github.com/user-attachments/assets/3fb51eee-e44d-4b93-a4a0-5f53a6347d87)

![image](https://github.com/user-attachments/assets/8c96b912-d1cd-44b4-ac76-8e0246cfd7c7)


## 결론

- point를 사용한 mask 추출은 성공적임
- box를 그려서 mask를 추출하는 것은 실패함
- 우선 맨 앞의 어린이를 대상으로 mask 추출, 뒤의 어린이도 추출해야 함
- sam2 다운로드가 안돼서 따로 코드를 빼서 다운로드 함
- sam2.1_hiera_large.pt 다운로드가 안돼서 공식에서 다운로드하여 드라이브에 붙여넣음
