# YOLOv8_Blink02
1. '자동차(CAR)'와 '자동차 신호등(Blink)'을 구별하는 YOLOv8 
2. Colab과 Local에서 실습함

![194556](https://github.com/user-attachments/assets/9d9e6892-abda-41de-b480-53fd52c0d0b3)
![195111](https://github.com/user-attachments/assets/d78fc904-692f-47c4-8166-b5e28666e293)

# result_mvideo_250116.ipynb 
- AONA Drive 저장됨
1. 공사현장에서 사용하는 안전장비를 object detection함
2. 이 코드에는 라벨의 크기를 조절할 수 있는 코드가 있음
   ```
   img = results[0].plot(line_width=2, font_size=10, pil=True)
3. 동영상으로 저장하여 압축 가능함
