# 딥러닝 기말 프로젝트

**[DeepLabV3++ Model 1](#DeepLabV3++-Model-1)** : branch [one_way] <br>
**[DeepLabV3++ Model 2](#DeepLabV3++-Model-2)** : branch [main]

## 실행환경
### 사용방법
> #### 환경설정
> 기본 환경 cuda 11.8 / 리눅스 20.04 
> 1. ```conda create -n ds python=3.9 -y```
> 2. ```conda activate ds```
> 3. ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```
> 4. ```pip install pandas opencv-python numpy tqdm matplotlib```
> 5. ```pip install albumentations timm```

## 각 경로 설정 방법
### Model 1
### Model 2
dualPrint.py 파일 <br>
`21줄 --- OUTPUT_DIR = "가중치 및 시각화 저장 폴더"` <br>
`204줄 --- setup_dataset_from_aihub('데이터 폴더 위치')`

## DeepLabV3++ Model 1
1. aihub_dataset을 `print_xception_model.py`과 같은 디렉토리에 download
2. `python print_xception_model.py`를 사용하여 실행

## DeepLabV3++ Model 2
1. aihub_dataset을 `print_xception_model.py`과 같은 디렉토리에 download
2. `python dualPrint.py`를 사용하여 실행
