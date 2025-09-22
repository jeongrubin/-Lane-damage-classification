# Damaged-Lane-Marking-Classification
차선 훼손 분류 프로젝트 

---

## 프로젝트 개요
도로 위 차선은 안전 운전의 기준선입니다.  
하지만 실제 주행 환경에서는 대형 차량의 압력, 날씨 변화, 제설제 같은 화학적 요인 때문에 차선이 쉽게 훼손됩니다.  

이러한 훼손은 **야간·우천 시 시야 확보를 어렵게 하고, 교통사고 위험을 높이는 주요 원인**이 됩니다.  
기존에는 **고가의 휘도 측정 장비**로 차선을 측정했지만, 데이터 수집·비용 문제로 한계가 있었습니다.  

➡ 따라서 본 프로젝트에서는 **차량 내부 카메라 + 딥러닝 분류 모델(CNN, ResNet)** 만으로도  
**차선 훼손 여부를 자동으로 분류할 수 있는지**를 탐구했습니다.  

---

## 프로젝트 목표
- 실제 도로 영상에서 추출한 차선 이미지를 **정상 / 훼손 정도별로 분류**
- Baseline(CNN) → 심화 모델(ResNet18) 성능 비교
- 클래스 불균형(A=220, B=102, C=94, D=13)을 고려한 학습 전략 탐색

---

## 데이터셋

| 클래스 | 설명 | 이미지 수 |
|--------|----------------------|---------|
| A | 정상 차선 (No damage) | 220 |
| B | 경미한 손상 (Minor damage) | 102 |
| C | 중간 손상 (Moderate damage) | 94 |
| D | 심각 손상 (Severe damage) | 13 |

- 총 이미지 수: **429장**
- 클래스 분포:
  - **A (220장)**
  - **B (102장)**
  - **C (94장)**
  - **D (13장)**
![Lane Damage](https://drive.google.com/uc?export=view&id=1kesqj-T0fz_wrxWlehSoPqKvt1Y8IfAg)

 **문제점**: 클래스 불균형이 심각 → 특히 D 클래스(훼손 심각)가 부족하여 모델 학습 어려움

---

## 전처리 & 데이터 증강
- **Resize**: 128×128 (모델 입력 크기 통일)
- **정규화**: 평균 0.5, 표준편차 0.5
- **증강 (학습 데이터 한정)**:
  - Random Horizontal Flip
  - Random Rotation (±10도)
  - Color Jitter

👉 하지만 단순 증강만으로 **클래스 D 부족 문제는 해결되지 않음**을 확인했습니다.  
 ➡ **데이터 확보 자체가 성능 향상에 더 중요**하다는 교훈을 얻음.

---

## 모델 설계
1. **CNN Baseline**
   - Conv → ReLU → Pooling 구조
   - Dropout 적용
   - 목적: baseline 성능 확인

2. **ResNet18**
   - Residual Block 구조 적용
   - 더 깊은 네트워크로 복잡한 패턴 학습
   - Transfer Learning 없이 scratch 학습


---

### CNN 예측 결과
![CNN Result](https://drive.google.com/uc?export=view&id=1TiX_B1yFDoT1uFVpJ8gtpz1ia07pnfUc)

---

### ResNet 예측 결과
![ResNet Result](https://drive.google.com/uc?export=view&id=1tH9gAR-adVaVll-gBnBepAmlsQmHBrGM)

---

## 성능 비교
- **CNN Baseline** → Test Accuracy: **93.48%**
- **ResNet18** → Test Accuracy: **95.65%**

---

## 느낀점
- 작은 데이터셋에서도 CNN만으로 높은 성능 확보 가능
- 하지만 **클래스 불균형 문제**는 단순 증강으로는 한계 → 데이터 확보 중요성 깨달음
- ResNet18이 baseline 대비 **더 안정적인 성능** 제공 확인

---

## 앞으로의 개선 방향
- **데이터 확충**: 다양한 도로 상황(비, 야간, 도심/고속도로) 데이터 확보
- **Transfer Learning**: 사전학습된 ResNet / EfficientNet 적용
- **Synthetic Data**: GAN을 활용해 훼손 차선 이미지 합성
- **Segmentation 확장**: 픽셀 단위로 훼손 영역까지 탐지 가능하게 발전

---

## 결론
이 프로젝트는 단순히 모델 정확도를 올리는 것을 넘어,  
➡**데이터 품질과 설계가 AI 성능에 얼마나 중요한가**를 깨달음
