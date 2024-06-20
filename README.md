
# Yolo
Yolo 모델 커스텀 데이터 학습

## 환경

<br>
<br>

## 설치 및 실행 환경 구성
1. 가상 환경 
    ```
    conda create -n yolo python=3.8
    conda activate yolo
    ```
<br>

2. pytorch 설치 
- Cuda 환경에 맞는 `torch`, `torchvision` 설치

<br>

3. `ultralytics` 설치
    ```
    pip install ultralytics
    ```
<br>

## 폴더 구성

```
{root}
├─ .gitignore
├─ READMD.md
├─ config
│  ├─ ...
├─ data
│  ├─ README.md
│  ├─ ...
├─ compare.py
├─ inference.py
├─ test.py
├─ torch2tensorrt.py
└─ train.py
```
- `config` 폴더에는 학습할 때 필요한 yaml 파일 만들어 저장
- `data` 폴더에는 학습을 위해 데이터를 전처리 하는 파일들이 저장되어 있는 폴더

<br>
<br>