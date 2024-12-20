#align_images_CEE

import cv2
import numpy as np
import os

def align_images_ecc(reference_image, target_image):
    """
    使用 ECC 方法對齊影像。
    :param reference_image: 參考影像 (灰階)
    :param target_image: 目標影像 (灰階)
    :return: 對齊後的目標影像
    """
    # 將影像轉為灰階
    reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    
    # 初始化變換矩陣 (3x3)
    warp_matrix = np.eye(3, 3, dtype=np.float32)

    # 設定停止條件
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)

    # 執行 ECC 影像註冊
    cc, warp_matrix = cv2.findTransformECC(reference_gray, target_gray, warp_matrix, cv2.MOTION_HOMOGRAPHY, criteria)
    
    # 套用變換至目標影像
    aligned_image = cv2.warpPerspective(target_image, warp_matrix, (reference_image.shape[1], reference_image.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return aligned_image

def crop_to_common_area(images):
    """
    找到所有影像的共同有效區域並裁切。
    :param images: 對齊後的影像列表
    :return: 裁切後的影像列表
    """
    combined_mask = None

    # 計算每張影像的有效區域 (非零像素)
    for image in images:
        mask = (image.sum(axis=2) > 0).astype(np.uint8)  # 假設非黑色區域為有效
        if combined_mask is None:
            combined_mask = mask
        else:
            combined_mask = cv2.bitwise_and(combined_mask, mask)

    if not combined_mask.any():
        raise ValueError("未找到有效的重疊區域，請檢查對齊參數或原始影像質量。")

    # 找到有效區域的邊界
    y_coords, x_coords = np.where(combined_mask)
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    # 裁切影像
    cropped_images = [image[y_min:y_max, x_min:x_max] for image in images]
    return cropped_images

def process_images(folder_path, output_folder):
    """
    主函數：讀取資料夾內所有影像，進行對齊並裁切。
    :param folder_path: 包含原始影像的資料夾路徑
    :param output_folder: 對齊和裁切後的影像輸出資料夾
    """
    # 讀取影像檔案
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    if not image_files:
        raise ValueError("資料夾中未找到影像檔案。")
    
    # 確保輸出資料夾存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 讀取參考影像
    reference_image = cv2.imread(os.path.join(folder_path, image_files[0]))
    aligned_images = [reference_image]  # 將參考影像加入列表

    # 對齊其他影像
    for image_file in image_files[1:]:
        target_image = cv2.imread(os.path.join(folder_path, image_file))
        aligned_image = align_images_ecc(reference_image, target_image)
        aligned_images.append(aligned_image)
    
    # 裁切所有影像到共同區域
    cropped_images = crop_to_common_area(aligned_images)

    # 保存裁切後的影像
    for i, cropped_image in enumerate(cropped_images):
        output_path = os.path.join(output_folder, f"aligned_cropped_{i+1}.png")
        cv2.imwrite(output_path, cropped_image)
        print(f"保存對齊並裁切的影像: {output_path}")

# 使用範例
input_folder = r"E:\Shitephen\Output log for FU hinoki from ARATA\FU hinoki 1st take (1-565)\jpge files\20241116_052823_FU general\postprocess"  # 修改為你的資料夾路徑
output_folder = r"E:\Shitephen\Output log for FU hinoki from ARATA\FU hinoki 1st take (1-565)\jpge files\20241116_052823_FU general\postprocess\aligned_cropped_images"  # 輸出資料夾名稱
# 如果要直接輸出在根目錄 直接用 "aligned_cropped_images"

process_images(input_folder, output_folder)
