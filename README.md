# 차선 손상 분류 및 세그멘테이션

차선 이미지를 손상 등급으로 분류하는 방법과 손상 영역을 픽셀 단위로 분할하는 방법을 실험했습니다.

---

## 프로젝트 개요

차선 손상 상태를 카메라 이미지로 판별하기 위해 분류와 세그멘테이션을 각각 적용했습니다. 분류 모델은 이미지 전체의 손상 등급을 예측하고, 세그멘테이션 모델은 손상이 나타난 위치를 마스크로 출력합니다.

1. **분류(Classification)**: 차선 이미지를 훼손 등급(A~D)으로 분류 (CNN Baseline → ResNet18)
2. **세그멘테이션(Segmentation)**: 같은 계열의 데이터에 픽셀 단위 손상 라벨을 추가로 확보해, 이미지 어디가 훼손됐는지 마스크로 예측 (U-Net)

분류 실험 이후 손상 위치와 면적을 확인할 수 있도록 세그멘테이션 실험을 추가했습니다.

---

## Part 1. 이미지 단위 분류 (Classification)

### 데이터셋

| 클래스 | 설명 | 이미지 수 |
|--------|----------------------|---------|
| A | 정상 차선 (No damage) | 220 |
| B | 경미한 손상 (Minor damage) | 102 |
| C | 중간 손상 (Moderate damage) | 94 |
| D | 심각 손상 (Severe damage) | 13 |

전체 데이터는 **429장**이며, D 클래스가 13장으로 클래스별 이미지 수의 차이가 큽니다.

![Sample Images from Each Class](assets/classification/class_samples.png)

### 전처리 & 데이터 증강
- Resize 128×128, 정규화(평균 0.5, 표준편차 0.5)
- 학습 데이터에 한해 Random Horizontal Flip / Random Rotation(±10°) / Color Jitter 적용
- D 클래스는 이미지 수가 적어 증강을 적용해도 데이터 불균형이 남았습니다.

### 모델 및 결과

| 모델 | 구조 | Test Accuracy |
|---|---|---:|
| CNN Baseline | Conv→ReLU→Pooling 3단 + Dropout | 93.48% |
| ResNet18 | Residual Block, scratch 학습(전이학습 없음) | 95.65% |

| CNN 예측 결과 | ResNet18 예측 결과 |
|---|---|
| ![CNN 예측](assets/classification/cnn_predictions.png) | ![ResNet18 예측](assets/classification/resnet_predictions.png) |

---

## Part 2. 픽셀 단위 손상 세그멘테이션 (Segmentation)

등급 분류 결과만으로는 손상 위치를 확인할 수 없습니다. 추가로 확보한 LabelMe 데이터에는 차선 영역과 개별 손상 부위가 폴리곤으로 표시되어 있어, 이를 이진 마스크로 변환해 U-Net을 학습했습니다.

### 데이터셋

| 클래스 | 설명 | 이미지 수 | 손상 폴리곤 수 |
|---|---|---:|---:|
| A | 정상 (손상 라벨 없음) | 250 | 0 |
| C | 중간 손상 | 84 | - |
| F | 심각 손상 | 42 | - |

총 **376장**, 손상 인스턴스 폴리곤 **740개**(C+F 이미지에만 존재). A클래스는 전부 배경뿐인 마스크로, 모델이 "손상 없음"을 정확히 배우는 데 사용됩니다.

클래스별 8:1:1로 층화 분할했습니다.

| Split | 전체 | A | C | F |
|---|---:|---:|---:|---:|
| Train | 302 | 200 | 68 | 34 |
| Val | 37 | 25 | 8 | 4 |
| Test | 37 | 25 | 8 | 4 |

### 모델: U-Net (4-level, from scratch)

256×256 입력, 인코더-디코더 각 4단, 채널 32→64→128→256→512(bottleneck). 손상 클래스가 이미지 내 소수 픽셀이라 `CrossEntropyLoss(weight=[1.0, 5.0])`로 손상 클래스에 가중치를 줬습니다. Random Flip/Rotation/ColorJitter 증강, Adam + Cosine 스케줄, 60 epoch.

### 결과 (test set, n=37, 픽셀 단위)

| Precision | Recall | F1-Score | IoU |
|---:|---:|---:|---:|
| 0.560 | 0.912 | 0.694 | 0.531 |

Recall이 Precision보다 높게 측정됐습니다. 손상 클래스에 5.0의 가중치를 적용해 미탐을 줄이도록 학습했지만, 그에 따라 과탐이 증가한 것으로 보입니다. 학습 데이터가 302장이고 사전학습 모델을 사용하지 않았다는 한계가 있습니다.

### Original / Ground Truth / Prediction 비교

![U-Net GT vs Prediction](assets/segmentation/unet_gt_compare.png)

일부 손상 영역은 정답 마스크와 비슷하게 예측했지만, 세 번째 행처럼 실제보다 넓은 영역을 손상으로 분류한 사례도 있습니다. 이는 Precision 0.560과 관련된 과탐 사례입니다. 마지막 행의 정상 차선에서는 손상 영역을 예측하지 않았습니다.

---

## 활용 방안

예측 마스크에서 손상 면적을 계산하면 도로 구간별 점검 우선순위를 정하는 자료로 활용할 수 있습니다.

1. 차량 블랙박스/CCTV 영상에서 주기적으로 차선 구간 crop을 추출
2. U-Net으로 각 crop의 손상 마스크를 예측 → `손상 픽셀 수 / 차선 전체 픽셀 수`를 "훼손도 점수"로 환산
3. GPS·구간 정보와 함께 누적해 도로 구간별 훼손도 지도를 생성
4. 훼손도 점수가 높은 구간부터 우선순위를 매겨 보수 일정에 반영

차량 카메라 영상을 이용해 점검이 필요한 구간을 먼저 선별하고, 이후 담당자가 해당 구간을 확인하는 보조 시스템으로 확장할 수 있습니다.

현재 결과는 제한된 데이터로 수행한 가능성 검토 단계입니다. 실제 적용 전에는 다양한 도로 환경에서의 일반화 성능, 처리 속도, 오탐률을 추가로 확인해야 합니다.

---

## 분류와 세그멘테이션 비교

- **분류**: 이미지마다 등급 라벨 하나가 필요하므로 라벨링 비용이 비교적 낮습니다.
- **세그멘테이션**: 손상 영역을 폴리곤으로 표시해야 하지만 손상의 위치와 면적을 계산할 수 있습니다.
- 이 프로젝트에서는 두 데이터셋의 클래스 체계도 다릅니다(분류: A/B/C/D 4단계, 세그멘테이션: A/C/F 3단계) — 동일한 촬영 소스에서 라벨링 방식만 바뀐 것이라 완전히 동일한 이미지 집합은 아닙니다.

---

## 코드 구성

```text
.
├── train_cnn_baseline.py         # Part 1: CNN Baseline 학습
├── train_resnet18.py             # Part 1: ResNet18 학습
├── models/unet.py                # Part 2: U-Net 아키텍처
├── train_lane_unet.py            # Part 2: U-Net 학습 + 평가
├── build_lane_seg_dataset.py     # LabelMe JSON → 학습용 image/mask 변환 및 분할
├── build_gt_comparison.py        # Original/GT/Prediction 비교 이미지 생성
├── lane_unet.pth                 # 학습된 U-Net 체크포인트 (재현 가능)
└── Line Damage_calssfiation.ipynb  # Part 1 원본 노트북(정리 전)
```

## 실행 방법

원본 이미지 데이터는 포함하지 않았습니다. 아래는 재현을 위한 최소 절차입니다.

```bash
pip install -r requirements.txt

# Part 2: LabelMe 라벨(raw_labels/{A,C,F}/*.png, *.json)로부터 데이터셋 구성
python build_lane_seg_dataset.py

# U-Net 학습 및 평가
python train_lane_unet.py

# Original/GT/Prediction 비교 이미지 생성
python build_gt_comparison.py
```

---

## 한계 및 정리

- 분류 모델은 90% 이상의 정확도를 기록했지만, D 클래스가 13장뿐이어서 클래스별 성능을 함께 확인할 필요가 있습니다.
- 세그멘테이션은 폴리곤 라벨이 필요하지만 손상의 위치와 면적을 확인할 수 있습니다.
- 세그멘테이션 데이터가 376장으로 여전히 작아 사전학습 백본을 붙이면(예: ResNet 인코더 U-Net) 추가 개선 여지가 있음
- 분류·세그멘테이션 데이터가 동일한 이미지 집합이 아니라, 두 모델의 성능을 직접 비교하기보다는 "같은 문제에 대한 서로 다른 접근"으로 해석하는 것이 정확함

## 앞으로의 개선 방향

- **사전학습 인코더 적용**: ResNet/EfficientNet을 U-Net 인코더로 사용해 작은 데이터셋에서의 일반화 향상
- **분류·세그멘테이션 데이터 통합**: 동일 이미지 집합에 두 라벨을 모두 붙여 직접 비교 가능하게 재구축
- **데이터 확충**: 다양한 도로 상황(비, 야간, 도심/고속도로) 데이터 확보
- **합성 데이터 검토**: 소수 클래스 보강을 위한 이미지 합성 방법 비교
