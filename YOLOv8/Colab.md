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
``` # 압축을 풀 zip이 있는 경로
# 압축을 푼 zip의 저장 경로 및 폴더 이름
import zipfile

with zipfile.ZipFile('/content/drive/MyDrive/Study/yolov8/yolob8_blinker02/Blinker_Data.zip') as target_file:
    target_file.extractall('/content/drive/MyDrive/Study/yolov8/yolob8_blinker02')
```

## 주어진 경로의 data.yaml 파일 내용 확인
``` !cat /content/drive/MyDrive/Study/yolov8/yolob8_blinker02/data.yaml ```

##  PyYAML 라이브러리 설치
