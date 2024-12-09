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

``` conda active loY3
``` 

``` pip install ipykernel
``` 

``` pip install jupyter notebook
``` 

``` python -m ipykernel install —user —name loY2 —display-name “ loY3”
``` 

``` pip install ultralytics
``` 

``` pip install roboflow
``` 
