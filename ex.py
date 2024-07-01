from mmseg.apis import inference_model, init_model
import mmcv
import cv2
import os
import numpy as np

# 설정 파일 및 체크포인트 파일 경로
config_file = '/mnt/4tb/hyundai/mmseg_hyundai/configs/patchnet/patchnet_hyundae_512x512.py'
checkpoint_file = '/mnt/4tb/hyundai/mmseg_hyundai/iter_20000.pth'

# 구성 파일 및 체크포인트 파일을 사용하여 모델을 초기화합니다.
model = init_model(config_file, checkpoint_file, device='cuda:0')

# 테스트할 이미지 경로
img_path = '/mnt/4tb/hyundai/data/wd/wd_sampling/CMR_GT_Frame-N2207413-230113154922-ADAS_DRV3-WD_CMR_FR-001-00000300.png'

# 이미지를 로드하고 리사이즈합니다.
img = mmcv.imread(img_path)
original_size = img.shape[:2]  # 원래 이미지 크기 저장 (height, width)
resized_img = cv2.resize(img, (512, 512))

# 이미지를 세그멘테이션하고 결과를 반환합니다.
result = inference_model(model, resized_img)

# 세그멘테이션 결과를 확인하고 리사이즈합니다.
seg_map = result.pred_sem_seg.data.cpu().numpy().squeeze()

# 세그멘테이션 결과 크기 출력
print("Segmentation result shape (before resize):", seg_map.shape)

# 결과 크기가 16x16인지 확인
if seg_map.shape == (16, 16):
    print("The segmentation result is 16x16.")
else:
    print("The segmentation result is not 16x16.")

# 16x16 배열의 고유값 및 빈도 출력
unique, counts = np.unique(seg_map, return_counts=True)
print("Unique values and their counts in the 16x16 result array:")
for u, c in zip(unique, counts):
    print(f"Value: {u}, Count: {c}")

# 각 패치의 크기 계산
patch_height = original_size[0] // 16
patch_width = original_size[1] // 16

# 원본 이미지를 복사하여 시각화용 이미지 생성
visualized_img = img.copy()

# 색상 및 투명도 설정
color_map = {
    1: (0, 0, 255, 128),  # 반투명한 빨간색
    2: (0, 0, 255)  # 불투명한 빨간색
}

# 각 패치를 원본 이미지에 매핑하여 색상 입히기
for i in range(16):
    for j in range(16):
        label = seg_map[i, j]
        if label in color_map:
            x_start = j * patch_width
            y_start = i * patch_height
            x_end = x_start + patch_width
            y_end = y_start + patch_height
            
            overlay = visualized_img[y_start:y_end, x_start:x_end].copy()
            overlay[:, :, :3] = color_map[label][:3]
            
            if label == 1:
                alpha = 0.5  # 반투명도 설정
                cv2.addWeighted(overlay, alpha, visualized_img[y_start:y_end, x_start:x_end], 1 - alpha, 0, visualized_img[y_start:y_end, x_start:x_end])
            else:
                visualized_img[y_start:y_end, x_start:x_end] = overlay

# 결과를 저장할 디렉토리 생성
save_dir = '/mnt/4tb/hyundai/mmseg_hyundai/results'
os.makedirs(save_dir, exist_ok=True)

# 시각화된 이미지를 저장합니다.
visualized_img_path = os.path.join(save_dir, 'visualized_result.png')
cv2.imwrite(visualized_img_path, visualized_img)

# 결과 이미지 출력 경로
print(f"Visualized result saved at: {visualized_img_path}")

