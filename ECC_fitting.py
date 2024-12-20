# ECC (Enhanced Correlation Coefficient) fitting
# 目前最好
# 加上了區域連通性分析
# 篩噪點在計算生長和分解量之後


import cv2
import os
import numpy as np
import pandas as pd

# 資料夾路徑
input_folder = r"E:\Shitephen\Output log for FU hinoki from ARATA\FU hinoki 1st take (1-565)\jpge files\20241116_052823_FU general\postprocess\aligned_cropped_images"
output_csv = r"E:\Shitephen\Output log for FU hinoki from ARATA\FU hinoki 1st take (1-565)\jpge files\20241116_052823_FU general\postprocess\results_ECC_CCA.csv"
output_visual_folder = os.path.join(input_folder, "visual_results_ECC_CCA")

os.makedirs(output_visual_folder, exist_ok=True)  # 確保輸出資料夾存在

# 獲取資料夾內的圖片名稱（按名稱排序）
image_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg') or f.endswith('.png')])

# 初始化結果列表
results = []

# 定義區域連通性分析的最小區域閾值
MIN_AREA_THRESHOLD = 500  # 可根據需要調整

# 遍歷相鄰圖片
for i in range(len(image_files) - 1):
    # 讀取相鄰兩張圖片
    img1_path = os.path.join(input_folder, image_files[i])
    img2_path = os.path.join(input_folder, image_files[i + 1])
    
    image1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # --- 1. 圖像配準 ---
    _, binary1 = cv2.threshold(image1, 127, 255, cv2.THRESH_BINARY)
    _, binary2 = cv2.threshold(image2, 127, 255, cv2.THRESH_BINARY)

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)
    try:
        cc, warp_matrix = cv2.findTransformECC(binary1, binary2, warp_matrix, cv2.MOTION_AFFINE, criteria)
        binary2_aligned = cv2.warpAffine(binary2, warp_matrix, (binary1.shape[1], binary1.shape[0]), flags=cv2.INTER_LINEAR)
    except cv2.error as e:
        print(f"Warning: Image alignment failed for {image_files[i]} and {image_files[i+1]} due to {e}")
        continue

    # --- 2. 計算面積 ---
    area1 = np.sum(binary1 == 255)
    area2_aligned = np.sum(binary2_aligned == 255)
    growth_area = np.sum((binary2_aligned == 255) & (binary1 == 0))
    decomposition_area = np.sum((binary1 == 255) & (binary2_aligned == 0))

    # --- 3. 區域連通性分析 (Connected Component Analysis) ---
    def apply_cca(binary_img, min_area):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
        filtered_binary = np.zeros_like(binary_img)
        for label in range(1, num_labels):  # 跳過背景（label 0）
            if stats[label, cv2.CC_STAT_AREA] >= min_area:
                filtered_binary[labels == label] = 255
        return filtered_binary

    growth_binary = ((binary2_aligned == 255) & (binary1 == 0)).astype(np.uint8) * 255
    decomposition_binary = ((binary1 == 255) & (binary2_aligned == 0)).astype(np.uint8) * 255

    growth_filtered = apply_cca(growth_binary, MIN_AREA_THRESHOLD)
    decomposition_filtered = apply_cca(decomposition_binary, MIN_AREA_THRESHOLD)

    # 計算過濾後的面積
    growth_area_filtered = np.sum(growth_filtered == 255)
    decomposition_area_filtered = np.sum(decomposition_filtered == 255)

    # --- 4. 視覺化生長與分解 ---
    visual_result = np.zeros((binary1.shape[0], binary1.shape[1], 3), dtype=np.uint8)
    visual_result[:, :, 1] = growth_filtered  # 生長用綠色
    visual_result[:, :, 2] = decomposition_filtered  # 分解用紅色

    visual_output_path = os.path.join(output_visual_folder, f"visual_{i+1}.jpg")
    cv2.imwrite(visual_output_path, visual_result)

    # --- 5. 儲存結果 ---
    results.append({
        'Image1': image_files[i],
        'Image2': image_files[i + 1],
        'Area1': area1,
        'Area2_Aligned': area2_aligned,
        'Growth_Area': growth_area,
        'Decomposition_Area': decomposition_area,
        'Growth_Area_Filtered': growth_area_filtered,
        'Decomposition_Area_Filtered': decomposition_area_filtered,
        'Visual_Result': visual_output_path
    })

# 將結果儲存為 CSV 檔案
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False, encoding='utf-8-sig')

print(f"處理完成，結果已儲存到 {output_csv}")
