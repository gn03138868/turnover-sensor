# turnover_sensor.py
# turnover sensor (ver 0.33 stable) 
# 加上了區域連通性分析
# 篩噪點在計算生長和分解量之後
# 使平移+旋轉+切變+縮放 試圖對準各種形狀的根
# 如果對齊失敗，會用簡單的平移+旋轉

# Three-in-one version
# Added regional connectivity analysis
# Noise filtering is performed after calculating growth and decomposition amounts
# Combine translation, rotation, shear, and scaling to align with various root shapes
# If alignment fails, use simple translation and rotation

# 三合一バージョン
# 領域の連結性分析を追加
# ノイズ除去は成長と分解量の計算後に行う
# 平行移動、回転、せん断、拡大縮小を組み合わせて、様々な形状の根に合わせる
# アライメントに失敗した場合は、単純な平行移動と回転を使用

import os
import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import queue
import time

def align_images(reference_image, target_image):
    """
    對齊影像，先嘗試平移、旋轉、切變與縮放。
    如果對齊失敗，則使用簡化的平移與旋轉模式。

    Align images. First attempt translation, rotation, shear, and scaling.
    If that fails, use a simplified translation and rotation mode.
    
    画像を整列させる。まず平行移動、回転、せん断、拡大縮小を試みる。
    失敗した場合は、簡略化された平行移動と回転モードを使用する。
    
    """
    a4_width, a4_height = 5100, 7021  # A4 尺寸 (600 DPI)

    # 調整輸入影像至 A4 大小
    # 入力画像をA4サイズに調整
    # Adjust input images to A4 size
    reference_image = cv2.resize(reference_image, (a4_width, a4_height))
    target_image = cv2.resize(target_image, (a4_width, a4_height))

    # 轉換為灰階
    # グレースケールに変換
    # Convert to greyscale
    reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

    # 初始化仿射變換矩陣
    # アフィン変換行列を初期化
    # Initialise affine transformation matrix
    warp_matrix_affine = np.eye(2, 3, dtype=np.float32)
    warp_matrix_euclidean = np.eye(2, 3, dtype=np.float32)

    # 設定停止條件
    # 停止条件を設定
    # Set termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)

    # 優先嘗試仿射變換
    # まずアフィン変換を試みる
    # First attempt affine transformation
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
        print("アフィン変換による整列に失敗しました。簡略化された平行移動と回転モードを試みます。")
        print("Affine transformation alignment failed. Attempting simplified translation and rotation mode.")
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
            print("平行移動と回転による整列に失敗しました。元の画像を返します。")
            print("Translation and rotation alignment failed. Returning original image.")
            aligned_image = target_image

    return aligned_image

def ensure_a4_size(image):
    """
    確保影像為A4大小，必要時進行填充或縮放。
    画像がA4サイズであることを確認し、必要に応じて埋め込みまたは拡大縮小を行う。
    Ensure the image is A4 size, padding or scaling if necessary.
    """
    a4_width, a4_height = 5100, 7021  # A4 (600 DPI)
    
    # 創建A4畫布
    # A4キャンバスを作成
    # Create A4 canvas
    a4_canvas = np.full((a4_height, a4_width, 3), (0, 0, 0), dtype=np.uint8)
    
    if image is not None:
        h, w = image.shape[:2]
        scale = min(a4_width / w, a4_height / h)

        # 調整影像大小
        # 画像サイズを調整
        # Adjust image size
        new_width = int(w * scale)
        new_height = int(h * scale)
        resized_image = cv2.resize(image, (new_width, new_height))

        # 計算居中位置
        # 中央に配置するための位置を計算
        # Calculate position for centring
        y_offset = (a4_height - new_height) // 2
        x_offset = (a4_width - new_width) // 2

        # 放置影像於畫布上
        # キャンバスに画像を配置
        # Place image on canvas
        a4_canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

    return a4_canvas

def process_images(input_folder, output_folder, progress_callback=None):
    """
    處理所有影像並確保輸出為A4大小。
    すべての画像を処理し、出力がA4サイズであることを確認する。
    Process all images and ensure output is A4 size.
    """
    os.makedirs(output_folder, exist_ok=True)

    image_files = sorted([f for f in os.listdir(input_folder) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    if not image_files:
        raise ValueError("資料夾中未找到影像檔案")
        raise ValueError("フォルダ内に画像ファイルが見つかりません。")
        raise ValueError("No image files found in the folder.")

    for i, image_file in enumerate(image_files, 1):
        target_path = os.path.join(input_folder, image_file)
        target_image = cv2.imread(target_path)
        
        if target_image is None:
            print(f"警告：無法讀取影像 {target_path}，跳過處理")
            print(f"警告：画像 {target_path} を読み込めません。処理をスキップします。")
            print(f"Warning: Unable to read image {target_path}. Skipping processing.")
            continue

        # 第一張圖直接儲存，其餘圖片與第一張對齊
        # 最初の画像はそのまま保存し、残りの画像は最初の画像に合わせて整列させる
        # Save the first image as is, align the rest to the first image
        if i == 1:
            reference_image = ensure_a4_size(target_image)
        else:
            target_image = ensure_a4_size(target_image)
            target_image = align_images(reference_image, target_image)
        
        output_path = os.path.join(output_folder, f"aligned_a4_{i}.png")
        cv2.imwrite(output_path, target_image)
        print(f"已儲存對齊影像: {output_path}")
        print(f"整列した画像を保存しました: {output_path}")
        print(f"Saved aligned image: {output_path}")
        
        # 更新進度回呼
        # 進捗コールバックを更新
        # Update progress callback
        if progress_callback:
            progress_callback((i / len(image_files)) * 100)

def process_ecc_fitting(input_folder, output_csv, output_visual_folder, progress_callback=None):
    """
    執行ECC擬合分析。
    ECC適合分析を実行する。
    Perform ECC fitting analysis.
    """
    os.makedirs(output_visual_folder, exist_ok=True)

    # 獲取資料夾內的圖片名稱（按名稱排序）
    # フォルダ内の画像名を取得（名前順にソート）
    # Get image names in the folder (sorted by name)
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png') or f.endswith('.jpg')])

    # 初始化結果列表
    # 結果リストを初期化
    # Initialise results list
    results = []

    # 定義區域連通性分析的最小區域閾值
    # 領域連結性分析の最小領域閾値を定義
    # Define minimum area threshold for regional connectivity analysis
    MIN_AREA_THRESHOLD = 500  
    # 可根據需要調整
    # 必要に応じて調整可能
    # Adjustable as needed

    # 遍歷相鄰圖片
    # 隣接する画像をループ処理
    # Loop through adjacent images
    for i in range(len(image_files) - 1):
        # 讀取相鄰兩張圖片
        # 隣接する2枚の画像を読み込む
        # Read two adjacent images
        img1_path = os.path.join(input_folder, image_files[i])
        img2_path = os.path.join(input_folder, image_files[i + 1])
        
        image1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        # --- 1. 圖像配準 ---
        # --- 1. 画像位置合わせ ---
        # --- 1. Image registration ---
        _, binary1 = cv2.threshold(image1, 127, 255, cv2.THRESH_BINARY)
        _, binary2 = cv2.threshold(image2, 127, 255, cv2.THRESH_BINARY)

        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)

        try:
            # ECC 對齊
            # ECC位置合わせ
            # ECC alignment
            cc, warp_matrix = cv2.findTransformECC(
                binary1, 
                binary2, 
                warp_matrix, 
                cv2.MOTION_AFFINE, 
                criteria, 
                None, 
                5  # gaussFiltSize
            )
            binary2_aligned = cv2.warpAffine(binary2, warp_matrix, (binary1.shape[1], binary1.shape[0]), flags=cv2.INTER_LINEAR)
        except cv2.error as e:
            print(f"ECC alignment failed for {image_files[i]} and {image_files[i+1]}, fallback to simple translation and rotation.")
            print(f"{image_files[i]}と{image_files[i+1]}のECC位置合わせに失敗しました。単純な平行移動と回転に切り替えます。")
            print(f"ECC alignment failed for {image_files[i]} and {image_files[i+1]}, falling back to simple translation and rotation.")    
            warp_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            center = (binary1.shape[1] // 2, binary1.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle=0, scale=1)
            binary2_aligned = cv2.warpAffine(binary2, rotation_matrix, (binary1.shape[1], binary1.shape[0]), flags=cv2.INTER_LINEAR)

        # 計算面積和分析
        # 面積計算と分析
        # Area calculation and analysis
        area1 = np.sum(binary1 == 255)
        area2_aligned = np.sum(binary2_aligned == 255)
        growth_area = np.sum((binary2_aligned == 255) & (binary1 == 0))
        decomposition_area = np.sum((binary1 == 255) & (binary2_aligned == 0))

        # 區域連通性分析
        # 領域連結性分析
        # Regional connectivity analysis
        def apply_cca(binary_img, min_area):
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
            filtered_binary = np.zeros_like(binary_img)
            for label in range(1, num_labels):
                if stats[label, cv2.CC_STAT_AREA] >= min_area:
                    filtered_binary[labels == label] = 255
            return filtered_binary

        growth_binary = ((binary2_aligned == 255) & (binary1 == 0)).astype(np.uint8) * 255
        decomposition_binary = ((binary1 == 255) & (binary2_aligned == 0)).astype(np.uint8) * 255

        growth_filtered = apply_cca(growth_binary, MIN_AREA_THRESHOLD)
        decomposition_filtered = apply_cca(decomposition_binary, MIN_AREA_THRESHOLD)

        growth_area_filtered = np.sum(growth_filtered == 255)
        decomposition_area_filtered = np.sum(decomposition_filtered == 255)

        # 視覺化
        # 視覚化
        # Visualisation
        visual_result = np.zeros((binary1.shape[0], binary1.shape[1], 3), dtype=np.uint8)
        visual_result[:, :, 1] = growth_filtered
        visual_result[:, :, 2] = decomposition_filtered

        visual_output_path = os.path.join(output_visual_folder, f"visual_{i+1}.jpg")
        cv2.imwrite(visual_output_path, visual_result)

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
        
        # 更新進度回呼
        # 進捗コールバックを更新
        # Update progress callback
        if progress_callback:
            progress_callback(((i + 1) / (len(image_files) - 1)) * 100)

    # 將結果儲存為 CSV 檔案
    # 結果をCSVファイルとして保存
    # Save results as a CSV file
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🌳 Root Turnover Sensor 根系回転センサー ver. 0.33")
        
        self.geometry("670x350")
        self.configure(bg="#f7f7f7")
        
        self.folder_path = ""
        self.output_folder = ""
        self.progress_var = tk.DoubleVar()
        self.elapsed_time_var = tk.StringVar(value="Elapsed Time: 0.00 s")
        
        self.queue = queue.Queue()
        self.create_widgets()
        
        # Add queue processing method to the main event loop
        self.after(100, self.process_queue)

    def create_widgets(self):
        title_label = tk.Label(
            self,
            text="Turnover Sensor",
            font=("Helvetica", 18, "bold"),
            fg="#333",
            bg="#f7f7f7",
        )
        title_label.pack(pady=10)

        self.select_button = ttk.Button(
            self, text="📁 Select Folder フォルダを選択", command=self.select_folder
        )
        self.select_button.pack(pady=10)

        self.align_button = ttk.Button(
            self, text="🖼️ Align Images 画像を整列させる", command=self.start_align_thread
        )
        self.align_button.pack(pady=10)

        self.fitting_button = ttk.Button(
            self, text="📊 ECC Fitting 擬合分析", command=self.start_fitting_thread
        )
        self.fitting_button.pack(pady=10)

        self.progress_bar = ttk.Progressbar(self, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(pady=15, fill="x", padx=20)

        self.time_label = tk.Label(
            self,
            textvariable=self.elapsed_time_var,
            font=("Arial", 12),
            fg="#555",
            bg="#f7f7f7",
        )
        self.time_label.pack(pady=10)
        
        self.status_label = tk.Label(self, text=" Developed by the Forest Utilisation Lab in collaboration with ARATA meeting members. Copyright © 2025. All rights reserved.", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)


    def select_folder(self):
        self.folder_path = filedialog.askdirectory(title="Select Folder")
        if self.folder_path:
            self.output_folder = os.path.join(self.folder_path, "aligned_images")
            os.makedirs(self.output_folder, exist_ok=True)

    def start_align_thread(self):
        if not self.folder_path:
            messagebox.showerror("Error", "Please select a folder.")
            return
        threading.Thread(target=self.start_align_process, daemon=True).start()

    def start_fitting_thread(self):
        if not self.folder_path:
            messagebox.showerror("Error", "Please select a folder.")
            return
        threading.Thread(target=self.start_fitting_process, daemon=True).start()

    def start_align_process(self):
        start_time = time.time()
        try:
            process_images(self.folder_path, self.output_folder, self.queue_progress)
            elapsed_time = time.time() - start_time
            self.queue.put(("done", elapsed_time))
        except Exception as e:
            self.queue.put(("error", str(e)))

    def start_fitting_process(self):
        start_time = time.time()
        try:
            output_csv = os.path.join(self.folder_path, "ECC_results.csv")
            visual_folder = os.path.join(self.folder_path, "ECC_visuals")
            process_ecc_fitting(self.output_folder, output_csv, visual_folder, self.queue_progress)
            elapsed_time = time.time() - start_time
            self.queue.put(("done", elapsed_time))
        except Exception as e:
            self.queue.put(("error", str(e)))

    def queue_progress(self, progress):
        self.queue.put(("progress", progress))

    def process_queue(self):
        try:
            while True:
                item = self.queue.get_nowait()
                if item[0] == "progress":
                    self.progress_var.set(item[1])
                elif item[0] == "done":
                    elapsed_time = item[1]
                    self.elapsed_time_var.set(f"Elapsed Time: {elapsed_time:.2f} s")
                    messagebox.showinfo("Success", "Processing complete!")
                elif item[0] == "error":
                    messagebox.showerror("Error", item[1])
                    return
        except queue.Empty:
            pass
        self.after(100, self.process_queue)

if __name__ == "__main__":
    app = Application()
    app.mainloop()
