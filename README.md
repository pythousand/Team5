# 감정 분석 모델을 활용한 음악 생성 프로그램<br/>Muisc Face

## :alarm_clock: 개발 기간: 5월 9일(목) ~ 6월 4일(수)

## 개발환경:
|IDE|프로그래밍<br/>언어|
|------|---|
|![Coalb](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)|![PYTHON](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)|

## :people_holding_hands: 멤버구성 및 역할
김윤성: EDA, Modeling, Model ensemble, Presentation

정강빈: EDA, Modeling, preprocessing, MusicFace 개발

천진원: EDA, Modeling, crop & seg, PPT

한재현: EDA, Modeling, 서기

## :robot: 모델 개발
> ### EDA
1. Json 파일을 DataFrame으로 변환 후 각종 정보를 확인했습니다.

각 이미지 데이터의 라벨링, 나이, 성별, 배경정보, Bounding Box 정보 등 각종 정보가 처리되어 있었습니다.
![image](https://github.com/DPTure/Team5/assets/155731578/3a5cbfc6-8756-4f12-a58d-1ef9186dd7bf)

2. 기초 통계 정보를 확인했습니다.

3. 간단한 시각화를 통해 데이터의 분포를 확인했습니다.
![image](https://github.com/DPTure/Team5/assets/155731578/695528c1-6612-4e9c-83d9-92f7e9ab0669)
![image](https://github.com/DPTure/Team5/assets/155731578/80930595-dadf-401e-a41a-486d11f59671)
![image](https://github.com/DPTure/Team5/assets/155731578/463d0ce4-4755-4953-adb3-fe37ae62503b)
![image](https://github.com/DPTure/Team5/assets/155731578/a30529cf-c8e3-406f-8069-e866085fec4a)
![image](https://github.com/DPTure/Team5/assets/155731578/984407c3-a7ea-4f16-bb7c-d51094314ade)


> ### Preprocessing
1. 사진 데이터 crop & crop+seg 작업을 진행했습니다.

crop


![image](https://github.com/DPTure/Team5/assets/155731578/3350fccc-2835-42a8-b102-d6a2d609bb3a)


crop & seg


![image](https://github.com/DPTure/Team5/assets/155731578/7ec94f04-92f7-486a-89d3-4d68ca533953)


2. array 변환, 차원 추가 및 resize 전처리 작업을 수행했습니다.

> ### Modeling & Model ensemble

ConvNeXt, ResNet v2, DenseNet, MobileNet v2/v3, Inception v3, BEIT, SWIN, YOLO v8 모델을 돌려보고 loss 및 acc를 측정하였고,
모델 앙상블도 진행하였습니다. 이 중 가장 높은 성능을 보인 BEIT를 최종 감정 분석 모델로 선정하였습니다.
