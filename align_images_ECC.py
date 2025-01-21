#align_images_ECC
#再次修正版本使輸出不亂跑
#不切變與縮放，只有旋轉和平移避免讓原圖失真

import cv2
import numpy as np
import os

def align_images_ecc(reference_image, target_image):
    """
    使用 ECC 方法對齊影像，確保輸出維持 A4 大小。
    只允許平移和旋轉，避免切變和縮放。
    :param reference_image: 參考影像 (A4 大小)
    :param target_image: 目標影像 (A4 大小)
    :return: 對齊後的目標影像 (A4 大小)
    """
    # 確保輸入影像為 A4 大小
    a4_width, a4_height = 4960, 7014  # A4 尺寸 (600 DPI)
    
    # 調整輸入影像至 A4 大小
    reference_image = cv2.resize(reference_image, (a4_width, a4_height))
    target_image = cv2.resize(target_image, (a4_width, a4_height))

    # 轉換為灰階
    reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

    # 初始化變換矩陣
    warp_matrix = np.eye(2, 3, dtype=np.float32)  # 2x3 矩陣，適用於平移和旋轉

    # 設定停止條件
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)

    try:
        # 執行 ECC 影像註冊
        cc, warp_matrix = cv2.findTransformECC(
            reference_gray, target_gray, warp_matrix, cv2.MOTION_EUCLIDEAN, 
            criteria, None, 5
        )

        # 套用變換並確保輸出為 A4 大小
        aligned_image = cv2.warpAffine(
            target_image, warp_matrix,
            (a4_width, a4_height),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)  # 黑色填充
        )
    except cv2.error:
        print("警告：對齊失敗，返回原始大小的目標影像")
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
    a4_canvas = np.full((a4_height, a4_width, 3), 255, dtype=np.uint8)
    
    if image is not None:
        # 計算縮放比例，保持原始比例
        h, w = image.shape[:2]
        scale_w = a4_width / w
        scale_h = a4_height / h
        scale = min(scale_w, scale_h)

        # 調整影像大小
        new_width = int(w * scale)
        new_height = int(h * scale)
        resized_image = cv2.resize(image, (new_width, new_height))

        # 計算居中位置
        y_offset = (a4_height - new_height) // 2
        x_offset = (a4_width - new_width) // 2

        # 將調整後的影像放置在A4畫布上
        a4_canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

    return a4_canvas

def process_images(folder_path, output_folder):
    """
    主函數：處理所有影像並確保輸出為A4大小。
    """
    # 建立輸出資料夾
    os.makedirs(output_folder, exist_ok=True)

    # 讀取所有影像檔案
    image_files = sorted([f for f in os.listdir(folder_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    if not image_files:
        raise ValueError("資料夾中未找到影像檔案")

    # 讀取並處理參考影像
    reference_path = os.path.join(folder_path, image_files[0])
    reference_image = cv2.imread(reference_path)
    if reference_image is None:
        raise ValueError(f"無法讀取參考影像: {reference_path}")

    # 確保參考影像為A4大小
    reference_image = ensure_a4_size(reference_image)
    
    # 儲存處理後的參考影像
    output_path = os.path.join(output_folder, f"aligned_a4_1.png")
    cv2.imwrite(output_path, reference_image)
    print(f"已儲存參考影像: {output_path}")

    # 處理其餘影像
    for i, image_file in enumerate(image_files[1:], 2):
        target_path = os.path.join(folder_path, image_file)
        target_image = cv2.imread(target_path)
        
        if target_image is None:
            print(f"警告：無法讀取影像 {target_path}，跳過處理")
            continue

        # 確保目標影像為A4大小
        target_image = ensure_a4_size(target_image)
        
        # 對齊影像
        aligned_image = align_images_ecc(reference_image, target_image)
        
        # 再次確保輸出為A4大小
        final_image = ensure_a4_size(aligned_image)
        
        # 儲存結果
        output_path = os.path.join(output_folder, f"aligned_a4_{i}.png")
        cv2.imwrite(output_path, final_image)
        print(f"已儲存對齊影像: {output_path}")

if __name__ == "__main__":
    input_folder = r"E:\Shitephen\Output log for FU hinoki from ARATA\FU hinoki 1st take (1-565)\jpge files\20241221_083815_3f\postprocess"
    output_folder = r"E:\Shitephen\Output log for FU hinoki from ARATA\FU hinoki 1st take (1-565)\jpge files\20241221_083815_3f\postprocess\aligned_cropped_images"
    process_images(input_folder, output_folder)
    #請改成自己想要的路徑

