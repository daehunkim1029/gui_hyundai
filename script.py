import os
import sys
from pathlib import Path
import mmcv
import cv2
import numpy as np
from mmseg.apis import init_model, inference_model


def open_directory(path):
    if sys.platform.startswith('darwin'):  # macOS
        os.system('open "{}"'.format(path))
    elif sys.platform.startswith('win'):  # Windows
        os.system('start "" "{}"'.format(path))
    elif sys.platform.startswith('linux'):  # Linux
        os.system('xdg-open "{}"'.format(path))
    else:
        print("Unsupported operating system.")

class MMSegWrapper:
    def __init__(self):
        self.model = None
        self.download_model()

    def download_model(self):
        config_file = 'configs/patchnet/0920/r34_1.py'
        #checkpoint_file = '/mnt/PatchModel/sota_for_gui.pth'
        #checkpoint_file = 'in_20images.pth'
        checkpoint_file = 'out_downlr.pth'
        self.model = init_model(config_file, checkpoint_file, device='cuda:0')

    def get_result(self, src):
        if self.model:
            try:
                result_dict = {}
                if isinstance(src, str):  # src가 파일 경로인 경우
                    ext = Path(src).suffix
                    save_dir = '/mnt/4tb/hyundai/data/val'
                    os.makedirs(save_dir, exist_ok=True)
                    dst_filename = os.path.join(save_dir, f'{Path(src).stem}_result{ext}')
                    img = mmcv.imread(src)
                elif isinstance(src, np.ndarray):  # src가 numpy 배열인 경우
                    img = src
                    dst_filename = None
                else:
                    raise Exception('Unsupported input type.')

                original_size = img.shape[:2]
                resized_img = cv2.resize(img, (512, 512))
                result = inference_model(self.model, resized_img)
                seg_map = result.pred_sem_seg.data.cpu().numpy().squeeze()

                result_dict['seg_map'] = seg_map
                unique, counts = np.unique(seg_map, return_counts=True)
                result_dict['unique_values'] = unique
                result_dict['counts'] = counts
                #print("Unique values and their counts in the 16x16 result array:")
                #for u, c in zip(unique, counts):
                #    print(f"Value: {u}, Count: {c}")

                patch_height = original_size[0] // 16
                patch_width = original_size[1] // 16

                visualized_img = img.copy()
                color_map = {
                    1: (180, 98, 0),  # 반투명한 빨간색
                    2: (95, 44, 0)  # 불투명한 빨간색
                }

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

                            alpha = 0.5 if label == 1 else 0.7  # 레이블 1은 30% 불투명, 레이블 2는 70% 불투명
                            cv2.addWeighted(overlay, alpha, visualized_img[y_start:y_end, x_start:x_end], 1 - alpha, 0, visualized_img[y_start:y_end, x_start:x_end])

                if dst_filename: 
                    #import pdb; pdb.set_trace() # 파일 경로가 주어졌을 때만 저장
                    cv2.imwrite(dst_filename, visualized_img)
                    result_dict['dst_filename'] = dst_filename
                    return dst_filename, result_dict
                else:
                    return visualized_img, result_dict
                    import pdb; pdb.set_trace()
            except Exception as e:
                raise Exception(e)
        else:
            raise Exception('You have to call download_model first.')

if __name__ == "__main__":
    # 테스트할 이미지 경로
    img_path = '/mnt/4tb/hyundai/data/val'
    
    # MMSegWrapper 객체 생성 및 모델 다운로드
    wrapper = MMSegWrapper()
    
    for img_path in Path(img_path).glob('*'):
        if img_path.suffix in ['.jpg', '.png', '.jpeg']:
            dst_filename, result_dict = wrapper.get_result(str(img_path))
            print(f"Segmentation result saved at: {dst_filename}")
