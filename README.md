# 딥러닝 기말 프로젝트

**[DeepLabV3++ Model 1]** : branch [one_way] <br>
**[DeepLabV3++ Model 2]** : branch [main]
---
## 실행환경
> #### 환경설정
> 기본 환경 cuda 11.8 / 리눅스 20.04 
> 1. ```conda create -n deeps python=3.9 -y```
> 2. ```conda activate deeps```
> 3. ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```
> 4. ```pip install pandas opencv-python numpy tqdm matplotlib```
> 5. ```pip install albumentations timm```
---
## 데이터 다운로드
[데이터 다운로드](https://drive.google.com/drive/folders/1us3Nwr37IuqZfPR8kkJspbQxkv5bx7Ac?usp=drive_link)

## 폴더 구조
```
Final_DeepSystem/
├── network/
│   ├── backbone/
│   │   ├── ...
│   │   └── xception.py
│   ├── modeling.py
│   └── _deeplab.py                <- model 1 구조 코드
├── print_xception_model.py        <- one_way branch model 1 학습 및 시각화 코드
├── dualPrint.py                   <- main branch model 2 학습 및 시각화 코드
└── CBAMDCNV2Deep.py               <- model 2 구조 코드
```

## 각 경로 설정 방법
### Model 1
print_xception_model.py 파일 <br>
`18줄 --- OUTPUT_DIR = "가중치 및 시각화 저장 폴더"` <br>
`194줄 --- setup_dataset_from_aihub('데이터 폴더 위치')`
### Model 2
dualPrint.py 파일 <br>
`21줄 --- OUTPUT_DIR = "가중치 및 시각화 저장 폴더"` <br>
`204줄 --- setup_dataset_from_aihub('데이터 폴더 위치')`

---
## DeepLabV3++ Model 1
1. `git clone --branch one_way https://github.com/choimagon/Final_DeepSystem.git`
2. `python print_xception_model.py`를 사용하여 실행

## DeepLabV3++ Model 2
1. `git clone https://github.com/choimagon/Final_DeepSystem.git` 
2. `python dualPrint.py`를 사용하여 실행

---
