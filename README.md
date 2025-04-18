# OBB Data Augmentation

이 프로젝트는 **OBB (Oriented Bounding Box)** 데이터를 대상으로  
**회전(Rotation)** 및 **랜덤 크롭(Random Crop)** 기반 **데이터 증강(Augmentation)** 기능을 제공합니다.

## Features

- **Rotation**: 입력 이미지와 OBB 라벨을 지정된 각도로 회전합니다.
- **Random Crop**: 이미지의 일부분을 랜덤하게 잘라내고, 이에 맞춰 OBB 라벨을 수정합니다.

## Requirements

- Python 3.x
- NumPy
- OpenCV (cv2)

## Usage

```bash
python main.py