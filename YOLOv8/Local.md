# 가상환경 설정
**주의**
- **가상환경 이름**과 **커널 이름**을 같게 해야 헷갈리지 않음
- **tensorflow**는 사용하지 않음

YOLOv8 개발환경 설치 (cuda 11.8 cndnn 8.7.0): <https://velog.io/@tjdwjdgus99/YOLOv8-%EA%B0%9C%EB%B0%9C%ED%99%98%EA%B2%BD-%EC%84%A4%EC%B9%98-cuda-11.8-cndnn-8.7.0/>

## CUDA 설치
NVIDIA Developer CUDA Toolkit 11.8 Downloads: <https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local/>

![스크린샷 2024-12-06 172638](https://github.com/user-attachments/assets/f2058a56-e7ff-4d52-9678-4c9e34cc8fb8)
기본값으로 설치

## 시스템 환경 변수 확인
윈도우 검색창에 **시스템 환경 변수 편집** 검색 → 환경 변수(N)… → **CUDA_PATH_V11_8 더블클릭**
![스크린샷 2024-12-06 172844](https://github.com/user-attachments/assets/730a6311-84a7-4905-8e17-13180a1b51d3)
**자동으로 환경변수가 설정되어 있을테니 확인만 하기**

## cuDNN 설치
1. <https://developer.nvidia.com/compute/cudnn/secure/8.6.0/local_installers/11.8/cudnn-windows-x86_64-8.6.0.163_cuda11-archive.zip/> 다운로드
- **엔비디아 계정이 있어야 다운로드 가능**
2. 밑의 경로에 가서 복사한 파일 붙여넣기
- C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
  ![스크린샷 2024-12-06 173244](https://github.com/user-attachments/assets/9658d705-3dd7-45f9-ad5e-86a5a829bf6c)
**다운로드한 파일 복사**
  ![스크린샷 2024-12-06 173333](https://github.com/user-attachments/assets/3cde5dbb-04dc-4014-b955-7dc08a32c8f6)
**붙여넣기**

## cmd창으로 연결 확인
![스크린샷 2024-12-06 173536](https://github.com/user-attachments/assets/db79288f-10a0-414e-b9fe-656bc15173ee)

```
cd C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\extras\demo_suite
```
![스크린샷 2024-12-06 173623](https://github.com/user-attachments/assets/3a8ed309-5aa9-4e85-afec-182dd5d2a1c7)


```
deviceQuery.exe
```
![스크린샷 2024-12-06 173636](https://github.com/user-attachments/assets/1888dace-49dd-4209-b128-0d5940f1e7b4)


![스크린샷 2024-12-06 173702](https://github.com/user-attachments/assets/40d33d0e-4e89-4616-ab64-08bac9da4e74)


# YOLOv8 설치
- python → 3.10.15
- ultralytics → 8.3.43
  
## 가상환경 생성
``` 
conda create -n loY3 python= 3.10.15
``` 

```
conda active loY3
``` 

```
pip install ipykernel
``` 

```
pip install jupyter notebook
``` 

```
python -m ipykernel install —user —name loY2 —display-name “ loY3”
``` 

```
pip install ultralytics
``` 

```
pip install roboflow
``` 
## 설치 확인

```
from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()
``` 
![스크린샷 2024-12-06 174204](https://github.com/user-attachments/assets/834ff3f1-0009-40ff-8e73-7298daa5a56e)
- CPU가 출력되고 있기에 GPU로 바꿔야 함

## YOLOv8 GPU 구성
- 아나콘다 창에 **nvidia-smi** 입력하여 버전 확인
```
nvidia-smi
```
![스크린샷 2024-12-06 174357](https://github.com/user-attachments/assets/4a04f2d0-7f66-4059-ad0a-bf057d923eec)

## pytorch 다운로드
<https://pytorch.org/get-started/locally/>
![스크린샷 2024-12-06 174430](https://github.com/user-attachments/assets/15b8c2fb-8c01-4b17-b837-9945a63a7f6e)

pytorch설치코드
```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

## cudatoolkit설치
```
conda install cudatoolkit
```

## 환경 설치 확인
```
import torch
torch.cuda.is_available()
```
![스크린샷 2024-12-06 174626](https://github.com/user-attachments/assets/ee6ce525-764f-4e83-addc-40788b0b799b)

## 오류 해결 코드
```
pip install --upgrade ultralytics
```

```
pip install numpy --upgrade
```

```
pip uninstall tensorflow-intel
```

# YOLOv8 Custom Dataset 실습
## Roboflow Custom Dataset 로컬에 다운로드
- Notebook이 실행되고 있는 현재 작업 디렉토리에 저장
- 데이터셋 다운로드 및 저장 경로 지정 
save_path = "/path/to/your/directory"
dataset = version.download("yolov8", path=save_path)

```
from roboflow import Roboflow
rf = Roboflow(api_key="**********")
project = rf.workspace("humantest").project("yolov8_blink02")
version = project.version(1)
dataset = version.download("yolov8")
```
![스크린샷 2024-12-06 180316](https://github.com/user-attachments/assets/9edfeb26-c6ab-4749-a4e7-ddcbf593122b)

## 주어진 경로의 data.yaml 파일 내용 확인
```
# 파일 경로 설정, 상대 경로 복사
file_path = 'yolov8_Blink02-1\data.yaml'

# 파일 내용 읽기 및 출력
with open(file_path, 'r') as file:
    content = file.read()
    print(content)
```
![스크린샷 2024-12-06 180348](https://github.com/user-attachments/assets/5386a5b7-b509-4ab0-8e8c-6a2e9345dd2f)

## 커스텀 데이터에 맞는 YAML 파일 만들기
```
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

```
import ultralytics

ultralytics.checks()
```
![스크린샷 2024-12-06 181512](https://github.com/user-attachments/assets/0677ec9e-89a3-406f-899f-b1270cdb93ec)

## Load a pre-trained model
```
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
```

```
print(type(model.names),len(model.names))

print(model.names)
```
![스크린샷 2024-12-06 181538](https://github.com/user-attachments/assets/cdeeaf8e-1c18-4332-98ac-1a3f05e463f0)


##  YOLOv8 커스텀 데이터 학습하기
```
model.train(data='yolov8_Blink02-1\data.yaml',epochs=100, patience = 32, imgsz=416),
```
![스크린샷 2024-12-06 181628](https://github.com/user-attachments/assets/c2ee5001-2947-4a28-8e51-2b64157c4648)


![스크린샷 2024-12-06 181645](https://github.com/user-attachments/assets/15747d21-1908-44b3-9e93-2ce2a4a0a52d)

##  학습된 YOLOv8 이용해서 테스트 이미지 예측
```
# train2
results = model.predict(source='yolov8_Blink02-1/test/images', save=True)
```
![스크린샷 2024-12-06 181712](https://github.com/user-attachments/assets/1968ee7d-94a2-4853-a887-1dbc8997a1b4)

## 테스트 이미지 제공하여 결과 확인
```
# train3
# 새로운 이미지 경로 설정
new_image_path = '194556.png'  # 여기에 이미지 경로를 입력하세요

# 이미지에서 사람 탐지
results = model.predict(source=new_image_path, save=True)

# 결과 시각화
for result in results:
    result.plot()  # 탐지된 결과를 시각화합니다.

```
![스크린샷 2024-12-06 181809](https://github.com/user-attachments/assets/802c14a1-4518-43cb-ba47-ddf7e5a71dde)
**train93**

![스크린샷 2024-12-06 181851](https://github.com/user-attachments/assets/94219ec6-4cc6-4b11-ba4d-14e822bf681a)


```
# train4
# 새로운 이미지 경로 설정
new_image_path = '195111.png'  # 여기에 이미지 경로를 입력하세요

# 이미지에서 사람 탐지
results = model.predict(source=new_image_path, save=True)

# 결과 시각화
for result in results:
    result.plot()  # 탐지된 결과를 시각화합니다.
```
![스크린샷 2024-12-06 181816](https://github.com/user-attachments/assets/13848a32-e235-472a-98f2-fe4be4482925)
**train94**

![스크린샷 2024-12-06 181912](https://github.com/user-attachments/assets/4fe665a8-c8c5-4424-a730-124f63883bdf)


### 오류 해결 폴더 구조

![image](https://github.com/user-attachments/assets/24879d28-8a33-4983-a408-a0fc9e33ddd3)

- **YOLOv8_Blink02_Local.ipynb** 위치 

         
![image](https://github.com/user-attachments/assets/2b34fe03-7f02-425d-9813-fce8f68f7fae)

- **data.yaml** 위치 


