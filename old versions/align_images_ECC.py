#align_images_ECC.py
#再次修正版本使輸出不亂跑
#會先仿射變換 (平移、旋轉、切變與縮放)
#失敗之後再用不切變與縮放，只有平移與旋轉（歐幾里得變換）

import cv2
import numpy as np
import os

def align_images(reference_image, target_image):
    """
    對齊影像，先嘗試平移、旋轉、切變與縮放。
    如果對齊失敗，則使用簡化的平移與旋轉模式。
    :param reference_image: 參考影像
    :param target_image: 目標影像
    :return: 對齊後的目標影像
    """
    a4_width, a4_height = 5100, 7021  # A4 尺寸 (600 DPI)

    # 調整輸入影像至 A4 大小
    reference_image = cv2.resize(reference_image, (a4_width, a4_height))
    target_image = cv2.resize(target_image, (a4_width, a4_height))

    # 轉換為灰階
    reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

    # 初始化仿射變換矩陣
    warp_matrix_affine = np.eye(2, 3, dtype=np.float32)
    warp_matrix_euclidean = np.eye(2, 3, dtype=np.float32)

    # 設定停止條件
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)

    # 優先嘗試仿射變換
    try:
        _, warp_matrix_affine = cv2.findTransformECC(
            reference_gray, target_gray, warp_matrix_affine, cv2.MOTION_AFFINE, 
            criteria, None, 5
        )
        aligned_image = cv2.warpAffine(
            target_image, warp_matrix_affine,
            (a4_width, a4_height),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
    except cv2.error:
        print("仿射變換對齊失敗，嘗試簡化的平移與旋轉模式")
        try:
            _, warp_matrix_euclidean = cv2.findTransformECC(
                reference_gray, target_gray, warp_matrix_euclidean, cv2.MOTION_EUCLIDEAN, 
                criteria, None, 5
            )
            aligned_image = cv2.warpAffine(
                target_image, warp_matrix_euclidean,
                (a4_width, a4_height),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )
        except cv2.error:
            print("平移與旋轉對齊失敗，返回原始影像")
            aligned_image = target_image

    return aligned_image

def ensure_a4_size(image):
    """
    確保影像為A4大小，必要時進行填充或縮放。
    :param image: 輸入影像
    :return: A4大小的影像
    """
    a4_width, a4_height = 4960, 7014  # A4 尺寸 (600 DPI)
    
    # 創建A4畫布
    a4_canvas = np.full((a4_height, a4_width, 3), (0, 0, 0), dtype=np.uint8)
    
    if image is not None:
        h, w = image.shape[:2]
        scale = min(a4_width / w, a4_height / h)

        # 調整影像大小
        new_width = int(w * scale)
        new_height = int(h * scale)
        resized_image = cv2.resize(image, (new_width, new_height))

        # 計算居中位置
        y_offset = (a4_height - new_height) // 2
        x_offset = (a4_width - new_width) // 2

        # 放置影像於畫布上
        a4_canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

    return a4_canvas

def process_images(folder_path, output_folder):
    """
    主函數：處理所有影像並確保輸出為A4大小。
    """
    os.makedirs(output_folder, exist_ok=True)

    image_files = sorted([f for f in os.listdir(folder_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    if not image_files:
        raise ValueError("資料夾中未找到影像檔案")

    reference_path = os.path.join(folder_path, image_files[0])
    reference_image = cv2.imread(reference_path)
    if reference_image is None:
        raise ValueError(f"無法讀取參考影像: {reference_path}")

    reference_image = ensure_a4_size(reference_image)
    output_path = os.path.join(output_folder, f"aligned_a4_1.png")
    cv2.imwrite(output_path, reference_image)
    print(f"已儲存參考影像: {output_path}")

    for i, image_file in enumerate(image_files[1:], 2):
        target_path = os.path.join(folder_path, image_file)
        target_image = cv2.imread(target_path)
        
        if target_image is None:
            print(f"警告：無法讀取影像 {target_path}，跳過處理")
            continue

        target_image = ensure_a4_size(target_image)
        aligned_image = align_images(reference_image, target_image)
        final_image = ensure_a4_size(aligned_image)
        
        output_path = os.path.join(output_folder, f"aligned_a4_{i}.png")
        cv2.imwrite(output_path, final_image)
        print(f"已儲存對齊影像: {output_path}")

if __name__ == "__main__":
    input_folder = r"E:\Shitephen\Output log for FU hinoki from ARATA\FU hinoki 1st take (1-565)\jpge files\20241116_052823_1b\postprocess\postprocess_test"
    output_folder = r"E:\Shitephen\Output log for FU hinoki from ARATA\FU hinoki 1st take (1-565)\jpge files\20241116_052823_1b\postprocess\postprocess_test\aligned_images"
    process_images(input_folder, output_folder)
