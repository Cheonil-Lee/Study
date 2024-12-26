# YOLOv8 Red_Zone 동영상_redZone2(완)
Dataset: https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow
Code: https://www.kaggle.com/code/hinepo/yolov8-inference-for-red-zone-application

pillow 절대 설치하지 말기

# 환경설정

생성 확인 

conda env list

가상환경 생성

conda create -n **redZone2**  python==3.10

환경 활성화

conda activate **redZone2**

---

pip install ipykernel

- 커널 충돌 → 누락된 ipykernel이 있을 수 있어서 설치를 해야함(conda)
    
    pip install certifi cycler filetype kiwisolver matplotlib numpy Pillow python-dotenv PyYAML requests tqdm urllib3
    

pip install jupyter notebook

python -m ipykernel install --user --name **redZone2** --display-name "**redZone2**"

**python: 3.10**

## YOLOv8 GPU 구성

아나콘다 창에 **nvidia-smi** 입력하여 버전 확인

<img width="562" alt="image" src="https://github.com/user-attachments/assets/f70ef1c9-29e5-4b72-88bc-72a57c138dda" />


**FileNotFoundError**: [WinError 2] 지정된 파일을 찾을 수 없습니다

→  설치 확인 `ffmpeg -version`

---

pip install ultralytics==8.1.29

`conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`

conda install cudatoolkit

conda install -c conda-forge ffmpeg

pip install opencv-python

---

```python
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
```

```python
import torch
torch.cuda.is_available()
```

True

```python
from IPython import display
display.clear_output()
import ultralytics
ultralytics.checks()
```
![image 1](https://github.com/user-attachments/assets/cb0bd837-35fb-44dd-8051-2f23a25c2633)


```python
! nvidia-smi -L
```

![image 2](https://github.com/user-attachments/assets/fe6662b7-ad2a-4893-b02d-7186c13de527)


## **Installs/Imports**

```python
import warnings
warnings.filterwarnings("ignore")

import os
import re
import glob
import yaml
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('Set3')

import IPython.display as display
from IPython.display import Video

from PIL import Image
import cv2

import torch

from ultralytics import YOLO
```

```python
import ultralytics
print(ultralytics.__version__)
```

8.1.29

## CFG**(구성, 설정) Configuration**

```python
class CFG:
    ### inference: use any pretrained or custom model
    # WEIGHTS = 'yolov8x.pt' # yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    
    # Yolov8_safety에서 만든 모델 사용
    WEIGHTS = 'C:/Users/YOON/Study/Red_zone/weights/best.pt'
    
    CONFIDENCE = 0.60
    CONFIDENCE_INT = int(round(CONFIDENCE * 100, 0))
    
    CLASSES_TO_DETECT = [0, 2, 4, 5, 7]  # Hardhat, NO-Hardhat, NO-Safety Vest, Person, Safety Vest, 내가 원하는 라벨 번호
    
    VERTICES_POLYGON = np.array([[200, 720], [0, 700], [500, 620], [990, 690], [820, 720]])

    EXP_NAME = 'ppe'

    ### 동영상 경로 수정
    VID_001 = 'C:\\Users\\YOON\\Study\\Red_zone\\archive\\example_video.mp4'
    
    ### choose filepath to make inference on (image or video)
    PATH_TO_INFER_ON = VID_001
    EXT = PATH_TO_INFER_ON.split('.')[-1]  # get file extension
    FILENAME_TO_INFER_ON = PATH_TO_INFER_ON.split('\\')[-1].split('.')[0]  # get filename

    ### paths
    ROOT_DIR = 'C:\\Users\\YOON\\Study\\Red_zone\\archive'
    OUTPUT_DIR = './'

```

```python
glob.glob(CFG.ROOT_DIR + '*')
```

['C:\\Users\\YOON\\Study\\Red_zone\\archive']

## **Image utils**

```python
def get_image_properties(image):
    if isinstance(image, str):
        # If image is a file path, read the image
        img = cv2.imread(image)
        if img is None:
            raise ValueError("Could not read image file")
    elif isinstance(image, np.ndarray):
        # If image is already a NumPy array, use it directly
        img = image
    else:
        raise ValueError("Input must be a file path or a NumPy array")

    # Get image properties
    properties = {
        "width": img.shape[1],
        "height": img.shape[0],
        "channels": img.shape[2] if len(img.shape) == 3 else 1,
        "dtype": img.dtype,
    }

    return properties
```

```python
def display_image(image, print_info = True, hide_axis = False):
    if isinstance(image, str):  # Check if it's a file path
        img = Image.open(image)
        plt.imshow(img)
    elif isinstance(image, np.ndarray):  # Check if it's a NumPy array
        image = image[..., ::-1]  # BGR to RGB
        img = Image.fromarray(image)
        plt.imshow(img);
    else:
        raise ValueError("Unsupported image format")

    if print_info:
        print('Type: ', type(img), '\n')
        print('Shape: ', np.array(img).shape, '\n')
        
    if hide_axis:
        plt.axis('off')
        
    plt.show()
```

## Video utils

```python
def get_video_properties(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    # Get video properties
    properties = {
        "fps": int(cap.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_seconds": int( cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) ),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
    }

    # Release the video capture object
    cap.release()

    return properties

```

```python
### testing function
video_properties = get_video_properties(CFG.VID_001)
video_properties
```

{'fps': 29,
'frame_count': 697,
'duration_seconds': 24,
'width': 1280,
'height': 720,
'codec': 877677894}

## **Video to make inference on**

```python
# conda install -c conda-forge ffmpeg
# ffmpeg -version
```

```python
OUT_VIDEO_NAME = './video_to_infer.mp4'

subprocess.run(
    [
        "ffmpeg",  "-i", CFG.PATH_TO_INFER_ON, "-crf",
        "18", "-preset", "veryfast", "-hide_banner", "-loglevel",
        "error", "-vcodec", "libx264", OUT_VIDEO_NAME
    ]
)

Video(data=OUT_VIDEO_NAME, embed=True, height=int(video_properties['height'] * 0.5), width=int(video_properties['width'] * 0.5))
```

![image 3](https://github.com/user-attachments/assets/d2cf79dd-fcdb-42e1-8108-2c22a202f949)


## **Adjust Red Zone**

```python
print('File path to make inference on: ', CFG.PATH_TO_INFER_ON)
print('File name to make inference on: ', CFG.FILENAME_TO_INFER_ON)
```

![image 4](https://github.com/user-attachments/assets/ce367173-b11b-49e7-b074-6b7f8e13fee7)


```python
cap = cv2.VideoCapture(CFG.PATH_TO_INFER_ON)

if not cap.isOpened():
    print("Error: Could not open video file.")
else:
    # Read the first frame
    ret, frame_test = cap.read()
    cap.release()

vertices_polygon = np.array([[200,720], [0,700], [500,620], [990,690], [820,720]]) # manually adjusted

cv2.polylines(frame_test, [vertices_polygon.reshape(-1, 1, 2)], True, (0, 0, 128), 4)

mod = frame_test.copy()
overlay = cv2.fillPoly(mod, pts = [vertices_polygon], color=(0, 0, 128))
background = frame_test.copy()

frame_test = cv2.addWeighted(
    src1 = background, # fisrt image
    alpha = 0.6, # first image weight
    src2 = overlay, # second image
    beta = 0.4, # second image weight
    gamma = 0.1, # scalar factor
    dst = overlay # output array shape
)
```

```python
display_image(frame_test)
```

![image 5](https://github.com/user-attachments/assets/98936d42-81c1-4b61-ade0-189fdd589422)

![image 6](https://github.com/user-attachments/assets/43cdf3b1-a3b9-4a52-8216-cc431472dfb7)


```python
img_properties = get_image_properties(frame_test)
img_properties
```

{'width': 1280, 'height': 720, 'channels': 3, 'dtype': dtype('uint8')}

## Model

```python
model = YOLO(CFG.WEIGHTS)
```

```python
print('Device: ', model.device)
```

Device:  cpu

```python
if torch.cuda.is_available():
    model.to('cuda:0')
else:
    print("CUDA device is not available. Running on CPU.")
```

```python
print('Device: ', model.device)
print('Weights: ', CFG.WEIGHTS)
```

![image 7](https://github.com/user-attachments/assets/9af2b053-95fe-4d72-981b-c75002923e55)


```python
# model.model
model.names
```

![image 8](https://github.com/user-attachments/assets/e73814da-7575-459d-976c-24c2ea13316b)


## **Inference**

```python
"""#%%time

results = model.predict(
    source=CFG.PATH_TO_INFER_ON,
    save=True,
    classes=CFG.CLASSES_TO_DETECT,
    conf=CFG.CONFIDENCE,
    save_txt=True,
    save_conf=True,
    device=0,
    stream=True,  # 메모리 문제 해결을 위한 설정
)
"""
```

```python
import os
print(os.path.exists(CFG.PATH_TO_INFER_ON))  # True가 출력되어야 합니다.
```

True

```python
import cv2
print(cv2.__version__)
```

4.10.0

```python
print(cv2.getBuildInformation())
```

![image 9](https://github.com/user-attachments/assets/6bb06923-7b84-484f-846f-c962dfcfe4cc)


```python
%%time

import cv2  # OpenCV 모듈 import

# 예측 수행
results = model.predict(
    source=CFG.PATH_TO_INFER_ON,
    save=True,
    classes=CFG.CLASSES_TO_DETECT,
    conf=CFG.CONFIDENCE,
    save_txt=True,
    save_conf=True,
    show=True,
    device=[0],
    # stream 주석처리하면 창과 함께 predict00 폴더 안에 동영상 생김
    # stream=True
)

# 예측이 끝난 후 모든 OpenCV 창 닫기
cv2.destroyAllWindows()
```

![image 10](https://github.com/user-attachments/assets/b39bbeed-0b6f-421a-9beb-24cf66cd20d1)


![image 11](https://github.com/user-attachments/assets/52a935cd-f710-4b53-a778-88ff1b381b9e)


## **Raw inference video**

```python
import glob
import subprocess
from IPython.display import Video

RAW_INFERENCE_VIDEO = glob.glob('archive/example_video.mp4')[0] # avi or mp4
OUT_VIDEO_NAME = './raw_inference.mp4'

subprocess.run(
    [
        "ffmpeg",  "-i", RAW_INFERENCE_VIDEO, "-crf",
        "18", "-preset", "veryfast", "-hide_banner", "-loglevel",
        "error", "-vcodec", "libx264", OUT_VIDEO_NAME
    ]
)

Video(data=OUT_VIDEO_NAME, embed=True, height=int(video_properties['height'] * 0.5), width=int(video_properties['width'] * 0.5))
```

![image 12](https://github.com/user-attachments/assets/5d84947e-faa1-47a7-86f2-9a48cf26189c)


```python
raw_inference_video_properties = get_video_properties(RAW_INFERENCE_VIDEO)
raw_inference_video_properties
```

{'fps': 29,
'frame_count': 697,
'duration_seconds': 24,
'width': 1280,
'height': 720,
'codec': 877677894}
