## Robolow CustomDataset 만들기
[YOLOv8_CustomData](https://www.notion.so/YOLOv8_CustomData-14ada9269783803a96dee92312429445?pvs=21) 참고

## Google Colab에서 Google Drive 마운트
```from google.colab import drive
drive.mount('/content/drive')
```

##  Google Colab에서 wget 명령어를 사용하여 Roboflow에서 만든 Custom Dataset 다운로드(Human_Data.zip)
```# Roboflow에서 만든 CustomDataset의 주소로 zip 다운로드
!wget -O /content/drive/MyDrive/Study/yolov8/yolob8_blinker02/Blinker_Data.zip https://app.roboflow.com/ds/7vZvGzeYrT?key=RnogJFLFwy
```
**!wget -O /content/drive/MyDrive/Study/yolov8/yolob8_blinker02/Blinker_Data.zip** 
→ 다운로드 받을 폴더의 경로 + 저장할 zip의 이름 지정

**https://app.roboflow.com/ds/7vZvGzeYrT?key=RnogJFLFwy**
→ CustomDataset의 Raw URL

# 데이터를 Colab으로 다운로드
```
# 압축을 풀 zip이 있는 경로
# 압축을 푼 zip의 저장 경로 및 폴더 이름
import zipfile

with zipfile.ZipFile('/content/drive/MyDrive/Study/yolov8/yolob8_blinker02/Blinker_Data.zip') as target_file:
    target_file.extractall('/content/drive/MyDrive/Study/yolov8/yolob8_blinker02')
```

## 주어진 경로의 data.yaml 파일 내용 확인
``` !cat /content/drive/MyDrive/Study/yolov8/yolob8_blinker02/data.yaml ```

##  PyYAML 라이브러리 설치
``` !pip install PyYAML ```

PyYAML
- YAML 파일을 읽고 쓰는 데 유용한 라이브러리
- 데이터 직렬화 및 구성을 쉽게 처리할 수 있게 도와줌

## 커스텀 데이터에 맞는 YAML 파일 만들기
```
import yaml

data = { 'train' : '/content/drive/MyDrive/Study/yolov8/yolob8_blinker02/train/images',
        'val' : '/content/drive/MyDrive/Study/yolov8/yolob8_blinker02/valid/images', # Make sure this path is correct
         'test' : '/content/drive/MyDrive/Study/yolov8/yolob8_blinker02/test/images',
         'names' : ['Blink', 'CAR', 'Truck'],
         'nc': 3}

with open('/content/drive/MyDrive/Study/yolov8/yolob8_blinker02/data.yaml', 'w') as f:
    yaml.dump(data, f)
```
## Install YOLOv8
``` !pip install ultralytics ```
```
import ultralytics

ultralytics.checks()
```

## Load a pre-trained model
```
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
```

```
print(type(model.names),len(model.names))

print(model.names)
```

## YOLOv8 커스텀 데이터 학습하기
```
model.train(data='/content/drive/MyDrive/Study/yolov8/yolob8_blinker02/data.yaml',epochs=100, patience = 32, imgsz=416),
```

## 학습된 YOLOv8 이용해서 테스트 이미지 예측
```
# train2
results = model.predict(source='/content/drive/MyDrive/Study/yolov8/yolov8_blinker/test/images', save=True)
```

## 예측한 테스트 이미지 Google Drive의 폴더에 저장
```
# train3
# 새로운 이미지 경로 설정
new_image_path = '/content/drive/MyDrive/Study/yolov8/yolob8_blinker02/194556.png'  # 여기에 이미지 경로를 입력하세요

# 이미지에서 사람 탐지
results = model.predict(source=new_image_path, save=True)

# 결과 시각화
for result in results:
    result.plot()  # 탐지된 결과를 시각화합니다.

```

```
# train4
# 새로운 이미지 경로 설정
new_image_path = '/content/drive/MyDrive/Study/yolov8/yolob8_blinker02/195111.png'  # 여기에 이미지 경로를 입력하세요

# 이미지에서 사람 탐지
results = model.predict(source=new_image_path, save=True)

# 결과 시각화
for result in results:
    result.plot()  # 탐지된 결과를 시각화합니다.

```
## Google Drive 폴더에 결과 이미지 저장
```
import shutil

# 복사할 폴더 경로
source_folder1 = '/content/runs/detect/train2'
source_folder2 = '/content/runs/detect/train3'
source_folder3 = '/content/runs/detect/train4'


# 구글 드라이브에 저장할 경로
destination_train2 = '/content/drive/MyDrive/Study/yolov8/yolob8_blinker02/train2'  # train2 복사 경로
destination_train3 = '/content/drive/MyDrive/Study/yolov8/yolob8_blinker02/train3'  # train3 복사 경로
destination_train4 = '/content/drive/MyDrive/Study/yolov8/yolob8_blinker02/train4'  # train2 복사 경로 # train3 복사 경로

# 폴더 복사
shutil.copytree(source_folder1, destination_train2)
shutil.copytree(source_folder2, destination_train3)
shutil.copytree(source_folder3, destination_train4)

import shutil
```





