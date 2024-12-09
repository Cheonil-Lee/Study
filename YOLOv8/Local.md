# 가상환경 설정
**주의**
- **가상환경 이름**과 **커널 이름**을 같게 해야 헷갈리지 않음
- **tensorflow**는 사용하지 않음
* YOLOv8 개발환경 설치 (cuda 11.8 cndnn 8.7.0): <https://velog.io/@tjdwjdgus99/YOLOv8-%EA%B0%9C%EB%B0%9C%ED%99%98%EA%B2%BD-%EC%84%A4%EC%B9%98-cuda-11.8-cndnn-8.7.0/>

## CUDA 설치
* NVIDIA Developer CUDA Toolkit 11.8 Downloads: <https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local/>

![스크린샷 2024-12-06 172638](https://github.com/user-attachments/assets/f2058a56-e7ff-4d52-9678-4c9e34cc8fb8)
기본값으로 설치

## 시스템 환경 변수 확인
윈도우 검색창에 **시스템 환경 변수 편집** 검색 → 환경 변수(N)… → **CUDA_PATH_V11_8 더블클릭**
![스크린샷 2024-12-06 172844](https://github.com/user-attachments/assets/730a6311-84a7-4905-8e17-13180a1b51d3)
**자동으로 환경변수가 설정되어 있을테니 확인만 하기**
