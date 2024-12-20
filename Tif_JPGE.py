#Tif to JPGE

import os
from PIL import Image

def batch_convert_to_jpeg(input_folder, output_folder, dpi=600, quality=100):
    """
    批量將指定資料夾中的圖片轉換為 JPEG 格式，並修改其 DPI 而不改變實際像素大小。

    Args:
        input_folder: 輸入資料夾路徑
        output_folder: 輸出資料夾路徑
        dpi: 輸出圖片的 DPI
        quality: JPEG 圖片的品質 (0-100)
    """

    os.makedirs(output_folder, exist_ok=True)  # 確保輸出資料夾存在

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.bmp', '.tiff', '.tif')):  # 支持更多格式
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_folder, file.replace(file.split('.')[-1], 'jpg'))

                try:
                    with Image.open(input_path) as im:
                        # 不改變圖像尺寸，直接設定 JPEG 的 DPI
                        im.save(output_path, 'JPEG', quality=quality, dpi=(dpi, dpi))
                        print(f"已成功轉換: {input_path} -> {output_path}")
                except Exception as e:
                    print(f"轉換失敗: {input_path}, 原因: {str(e)}")

# 替換成您的實際路徑
input_folder = r"Z:\RF for bamboo\Root for Hirano sensei and Ohashi sensei\Ryukoku\RC3-3f"
output_folder = r"Z:\RF for bamboo\Root for Hirano sensei and Ohashi sensei\Ryukoku\RC3-3f\jpge files"

batch_convert_to_jpeg(input_folder, output_folder)