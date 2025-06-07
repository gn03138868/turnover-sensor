# turnover_sensor_044.py
# 自動選擇最佳對齊方法 / 最適な整列方法の自動選択 / Automatic best alignment method selection
# 區域連通分析 / 領域の連結性分析 / Regional connectivity analysis
# 估計生長與分解量後移除噪點 / ノイズ除去は成長と分解量の計算後に行う / Noise filtering after growth/decomposition calculation
# 平移+旋转+切变+缩放對齊 / 平行移動+回転+せん断+拡大縮小で整列 / Alignment with translation+rotation+shear+scaling
# 增加OpticalFlow, Homography, 和bUnwarpJ非剛性配準 / OpticalFlow, Homography,と bUnwarpJ非剛性レジストレーション / OpticalFlow, Homography, and bUnwarpJ non-rigid registration

import os
import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import queue
import time
from skimage.morphology import remove_small_objects
import imagej
import tempfile
import tifffile
from scyjava import jimport

# 強制 NumPy 使用標準 Python 標量
np.set_printoptions(legacy='1.13')
pd._no_nep50_warning = True

# 安全整數轉換
def safe_int(value):
    """安全轉為 int"""
    if hasattr(value, 'item'):
        return value.item()
    return int(value)

# 強制 NumPy 標示轉換行為
def convert_numpy_scalars(value):
    if isinstance(value, np.generic):
        return value.item()
    elif isinstance(value, list):
        return [convert_numpy_scalars(v) for v in value]
    elif isinstance(value, tuple):
        return tuple(convert_numpy_scalars(v) for v in value)
    elif isinstance(value, dict):
        return {k: convert_numpy_scalars(v) for k, v in value.items()}
    return value

# GPU 檢測
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    cv2.cuda.setDevice(0)
    USE_CUDA = True
else:
    USE_CUDA = False

FIJI_PATH = ""   # 全局 Fiji 路徑
ij = None        # ImageJ 實例

def set_fiji_path():
    """設置 Fiji 路徑"""
    global FIJI_PATH
    path = filedialog.askdirectory(title="Select Fiji.app Directory")
    if path:
        FIJI_PATH = path
        messagebox.showinfo("Fiji Path Set", f"Fiji path set to: {FIJI_PATH}")

def initialize_imagej():
    """初始化 ImageJ/Fiji"""
    global ij
    if ij is not None:
        return ij
    if not FIJI_PATH:
        raise ValueError("Fiji path not set. Please set Fiji path first.")
    try:
        ij = imagej.init(FIJI_PATH, mode='interactive')
        print("ImageJ initialized successfully")
        return ij
    except Exception as e:
        raise RuntimeError(f"Failed to initialize ImageJ: {str(e)}")

def align_with_bunwarpj(reference_image, target_image, a4_width, a4_height):
    """使用 bUnwarpJ 進行非剛性配準"""
    try:
        ij_instance = initialize_imagej()

        with tempfile.TemporaryDirectory() as temp_dir:
            ref_path = os.path.join(temp_dir, "reference.tif")
            target_path = os.path.join(temp_dir, "target.tif")

            # 先保存為灰度 tif，供 bUnwarpJ 計算變形場
            tifffile.imwrite(ref_path, cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY))
            tifffile.imwrite(target_path, cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY))

            IJ = jimport('ij.IJ')
            ImagePlus = jimport('ij.ImagePlus')
            bUnwarpJ = jimport('bunwarpj.bUnwarpJ_')

            ref_imp = IJ.openImage(ref_path)
            target_imp = IJ.openImage(target_path)
            if ref_imp is None or target_imp is None:
                raise RuntimeError("Failed to load images in ImageJ")

            # bUnwarpJ 參數
            registration_mode = 0
            img_subsample_factor = 0
            min_scale_deformation = 0
            max_scale_deformation = 2
            divWeight = 0.1
            curlWeight = 0.1
            landmarkWeight = 0.0
            imageWeight = 1.0
            consistencyWeight = 10.0
            stopThreshold = 0.01

            transformation = bUnwarpJ.computeTransformationBatch(
                target_imp, ref_imp, None, None,
                registration_mode, img_subsample_factor,
                min_scale_deformation, max_scale_deformation,
                divWeight, curlWeight, landmarkWeight,
                imageWeight, consistencyWeight, stopThreshold
            )
            if transformation is None:
                raise RuntimeError("bUnwarpJ registration failed")

            transformed_imp = transformation.transform(target_imp)
            transformed_array = np.array(transformed_imp.getProcessor().getPixels(), dtype=np.uint8)
            h, w = target_image.shape[:2]
            transformed_array = transformed_array.reshape((h, w))
            aligned = cv2.cvtColor(transformed_array, cv2.COLOR_GRAY2BGR)

            # 統一使用中心裁切到 A4 尺寸
            return center_crop_to_a4(aligned, a4_width, a4_height)

    except Exception as e:
        print(f"bUnwarpJ alignment failed: {str(e)}")
        return center_crop_to_a4(target_image, a4_width, a4_height)


def write_uint8_png(src_tif, dst_png):
    """Read any bit-depth TIFF and write 8-bit PNG, preserving white points for masks"""
    arr = tifffile.imread(src_tif)
    if arr.dtype == np.uint8:
        cv2.imwrite(dst_png, arr)
        return
    if np.issubdtype(arr.dtype, np.floating) and float(arr.max()) <= 1.0:
        arr8 = (arr * 255).astype(np.uint8)
        cv2.imwrite(dst_png, arr8)
        return
    if np.issubdtype(arr.dtype, np.integer) and float(arr.max()) <= 255:
        arr8 = arr.astype(np.uint8)
        cv2.imwrite(dst_png, arr8)
        return
    arrf = arr.astype(np.float32)
    arr_min, arr_max = float(arrf.min()), float(arrf.max())
    if arr_max > arr_min:
        arr8 = ((arrf - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
    else:
        arr8 = np.zeros_like(arrf, dtype=np.uint8)
    cv2.imwrite(dst_png, arr8)

def preprocess_first_image(image, target_w, target_h):
    """Crop or pad around image center to target size"""
    h, w = image.shape[:2]
    crop_w = min(w, target_w)
    crop_h = min(h, target_h)
    x1 = (w - crop_w) // 2
    y1 = (h - crop_h) // 2
    region = image[y1:y1+crop_h, x1:x1+crop_w]
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - crop_w) // 2
    y_offset = (target_h - crop_h) // 2
    canvas[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w] = region
    return canvas


def center_crop_to_a4(image, a4_width, a4_height):
    """Center crop/pad image to exact A4 size with black padding"""
    h, w = image.shape[:2]
    
    # Create black canvas of A4 size
    canvas = np.zeros((a4_height, a4_width, 3), dtype=np.uint8)
    
    # Calculate crop dimensions (don't exceed A4 size)
    crop_h = min(h, a4_height)
    crop_w = min(w, a4_width)
    
    # Calculate crop start position (center of original image)
    start_y = (h - crop_h) // 2
    start_x = (w - crop_w) // 2
    
    # Crop from center of original image
    cropped = image[start_y:start_y + crop_h, start_x:start_x + crop_w]
    
    # Calculate placement position (center of canvas)
    canvas_y = (a4_height - crop_h) // 2
    canvas_x = (a4_width - crop_w) // 2
    
    # Place cropped image in center of canvas
    canvas[canvas_y:canvas_y + crop_h, canvas_x:canvas_x + crop_w] = cropped
    
    return canvas

def align_with_optical_flow(reference_image, target_image, a4_width, a4_height):
    """Non-rigid registration using dense optical flow (Farneback)"""
    ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    tgt_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        ref_gray, tgt_gray, None,
        pyr_scale=0.7, 
        levels=2, 
        winsize=21,
        iterations=5, 
        poly_n=7, 
        poly_sigma=1.5, 
        flags=0
    )
    h, w = target_image.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = x + flow[..., 0]
    map_y = y + flow[..., 1]
    aligned = cv2.remap(target_image, map_x, map_y, cv2.INTER_LINEAR)
    
    # 統一使用中心裁切到 A4 尺寸
    return center_crop_to_a4(aligned, a4_width, a4_height)

def align_with_homography(reference_image, target_image, a4_width, a4_height):
    """Perspective transformation using homography"""
    ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    tgt_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(tgt_gray, None)
    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
        return center_crop_to_a4(target_image, a4_width, a4_height)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    if len(matches) < 4:
        return center_crop_to_a4(target_image, a4_width, a4_height)
    src_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return center_crop_to_a4(target_image, a4_width, a4_height)
    
    # 先變形到參考圖像尺寸，再統一裁切
    h_ref, w_ref = reference_image.shape[:2]
    aligned = cv2.warpPerspective(target_image, H, (w_ref, h_ref))
    return center_crop_to_a4(aligned, a4_width, a4_height)


def evaluate_alignment_quality(reference_image, aligned_image):
    """Evaluate alignment via SSIM + Normalized Cross Correlation"""
    ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    aligned_gray = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
    ref_f = ref_gray.astype(np.float32)
    aligned_f = aligned_gray.astype(np.float32)

    mu1 = cv2.GaussianBlur(ref_f, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(aligned_f, (11, 11), 1.5)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(ref_f * ref_f, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(aligned_f * aligned_f, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(ref_f * aligned_f, (11, 11), 1.5) - mu1_mu2

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    ssim_score = np.mean(ssim_map)

    ref_norm = ref_f - np.mean(ref_f)
    aligned_norm = aligned_f - np.mean(aligned_f)
    ncc = np.sum(ref_norm * aligned_norm) / (np.sqrt(np.sum(ref_norm**2)) * np.sqrt(np.sum(aligned_norm**2)) + 1e-10)

    combined_score = 0.7 * ssim_score + 0.3 * ncc
    return combined_score, ssim_score, ncc

def try_all_alignment_methods(reference_image, target_image, a4_width, a4_height, progress_callback=None):
    """嘗試所有對齊方法並選擇最佳結果"""
    if USE_CUDA:
        gpu_ref = cv2.cuda_GpuMat()
        gpu_ref.upload(reference_image)
        gpu_tgt = cv2.cuda_GpuMat()
        gpu_tgt.upload(target_image)
        ref_gray = cv2.cuda.cvtColor(gpu_ref, cv2.COLOR_BGR2GRAY).download()
        tgt_gray = cv2.cuda.cvtColor(gpu_tgt, cv2.COLOR_BGR2GRAY).download()
    else:
        ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        tgt_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

    methods = ["Euclidean", "Affine", "OpticalFlow", "Homography"]
    if FIJI_PATH:
        methods.append("bUnwarpJ")

    results = []
    scores = []
    for method in methods:
        try:
            if progress_callback:
                progress_callback(0, f"Trying {method}...")
                
            if method == "Euclidean":
                warp_affine = np.eye(2, 3, dtype=np.float32)
                _, warp_matrix = cv2.findTransformECC(
                    ref_gray, tgt_gray,
                    warp_affine, cv2.MOTION_EUCLIDEAN,
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10), None, 5
                )
                h_ref, w_ref = reference_image.shape[:2]
                aligned = cv2.warpAffine(
                    target_image, warp_matrix,
                    (w_ref, h_ref),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
                )
                aligned = center_crop_to_a4(aligned, a4_width, a4_height)
                
            elif method == "Affine":
                warp_affine = np.eye(2, 3, dtype=np.float32)
                _, warp_matrix = cv2.findTransformECC(
                    ref_gray, tgt_gray,
                    warp_affine, cv2.MOTION_AFFINE,
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10), None, 5
                )
                h_ref, w_ref = reference_image.shape[:2]
                aligned = cv2.warpAffine(
                    target_image, warp_matrix,
                    (w_ref, h_ref),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
                )
                aligned = center_crop_to_a4(aligned, a4_width, a4_height)
                
            elif method == "OpticalFlow":
                aligned = align_with_optical_flow(reference_image, target_image, a4_width, a4_height)
            elif method == "Homography":
                aligned = align_with_homography(reference_image, target_image, a4_width, a4_height)
            elif method == "bUnwarpJ":
                aligned = align_with_bunwarpj(reference_image, target_image, a4_width, a4_height)
            else:
                continue

            score, ssim, ncc = evaluate_alignment_quality(reference_image, aligned)
            results.append((method, aligned, score, ssim, ncc))
            scores.append(score)
            if progress_callback:
                progress_callback(0, f"{method}: Score={score:.4f}")
        except Exception as e:
            print(f"{method} failed: {e}")
            if progress_callback:
                progress_callback(0, f"{method} failed: {str(e)[:50]}")
            continue

    if not results:
        return center_crop_to_a4(target_image, a4_width, a4_height), "None", 0.0, 0.0, 0.0

    best_idx = np.argmax(scores)
    best_method, best_aligned, best_score, best_ssim, best_ncc = results[best_idx]
    if progress_callback:
        progress_callback(0, f"Best: {best_method} (Score: {best_score:.4f})")
    return best_aligned, best_method, best_score, best_ssim, best_ncc
    

def align_single_method(reference_image, target_image, method, a4_width, a4_height):
    """使用指定方法進行單一對齊"""
    if USE_CUDA:
        gpu_ref = cv2.cuda_GpuMat()
        gpu_ref.upload(reference_image)
        gpu_tgt = cv2.cuda_GpuMat()
        gpu_tgt.upload(target_image)
        ref_gray = cv2.cuda.cvtColor(gpu_ref, cv2.COLOR_BGR2GRAY).download()
        tgt_gray = cv2.cuda.cvtColor(gpu_tgt, cv2.COLOR_BGR2GRAY).download()
    else:
        ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        tgt_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

    try:
        if method == "Euclidean":
            warp_affine = np.eye(2, 3, dtype=np.float32)
            _, warp_matrix = cv2.findTransformECC(
                ref_gray, tgt_gray,
                warp_affine, cv2.MOTION_EUCLIDEAN,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10), None, 5
            )
            h_ref, w_ref = reference_image.shape[:2]
            aligned = cv2.warpAffine(
                target_image, warp_matrix,
                (w_ref, h_ref),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )
            return center_crop_to_a4(aligned, a4_width, a4_height)
            
        elif method == "Affine":
            warp_affine = np.eye(2, 3, dtype=np.float32)
            _, warp_matrix = cv2.findTransformECC(
                ref_gray, tgt_gray,
                warp_affine, cv2.MOTION_AFFINE,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10), None, 5
            )
            h_ref, w_ref = reference_image.shape[:2]
            aligned = cv2.warpAffine(
                target_image, warp_matrix,
                (w_ref, h_ref),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )
            return center_crop_to_a4(aligned, a4_width, a4_height)
            
        elif method == "OpticalFlow":
            return align_with_optical_flow(reference_image, target_image, a4_width, a4_height)
        elif method == "Homography":
            return align_with_homography(reference_image, target_image, a4_width, a4_height)
        elif method == "bUnwarpJ":
            return align_with_bunwarpj(reference_image, target_image, a4_width, a4_height)
        else:
            return center_crop_to_a4(target_image, a4_width, a4_height)

    except Exception as e:
        print(f"{method} alignment failed: {e}")
        return center_crop_to_a4(target_image, a4_width, a4_height)



def process_images_with_method(input_folder, output_folder, a4_width, a4_height, selected_method, progress_callback=None):
    """使用指定方法處理圖像"""
    os.makedirs(output_folder, exist_ok=True)
    files = sorted([f for f in os.listdir(input_folder)
                    if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))])
    if not files:
        raise ValueError("No image files found in folder")

    results_log = []
    for i, fname in enumerate(files, start=1):
        img = cv2.imread(os.path.join(input_folder, fname))
        if img is None:
            continue

        # 對於第一張小的圖片進行特殊處理
        if i == 1 and img.size < 100000:
            if progress_callback:
                progress_callback(0, f"檢測到小體積檔案: {fname}, 應用特殊處理")
            temp_img = img.copy()
            lab = cv2.cvtColor(temp_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            temp_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            temp_img = cv2.resize(temp_img, (a4_width, a4_height), interpolation=cv2.INTER_CUBIC)
            img = temp_img

        if progress_callback:
            progress_callback((i/len(files))*50, f"Processing {fname}")

        if i == 1:
            ref = preprocess_first_image(img, a4_width, a4_height)
            out = ref
            used_method = "Reference"
            score = ssim = ncc = 1.0
        else:
            resized = center_crop_to_a4(img, a4_width, a4_height)
            if selected_method == "Auto":
                out, used_method, score, ssim, ncc = try_all_alignment_methods(
                    ref, resized, a4_width, a4_height, progress_callback)
            else:
                out = align_single_method(ref, resized, selected_method, a4_width, a4_height)
                used_method = selected_method
                score, ssim, ncc = evaluate_alignment_quality(ref, out)

        output_filename = f"aligned_a4_{i:02d}.png"
        cv2.imwrite(os.path.join(output_folder, output_filename), out)

        results_log.append({
            'Image': fname,
            'Output': output_filename,
            'Selected_Method': selected_method,
            'Used_Method': used_method,
            'Combined_Score': score,
            'SSIM': ssim,
            'NCC': ncc
        })

        if progress_callback:
            progress_callback((i/len(files))*100,
                            f"Saved {output_filename} (Method: {used_method}, Score: {score:.4f})")

    results_df = pd.DataFrame(results_log)
    results_df.to_csv(os.path.join(output_folder, "alignment_results.csv"), index=False)
    return results_df

def process_ecc_fitting(input_folder, output_csv, output_visual_folder, a4_width, a4_height, progress_callback=None):
    """ECC 擬合處理與分析 - 修復版本"""
    os.makedirs(output_visual_folder, exist_ok=True)
    
    # 檢查文件夾有沒有在
    if not os.path.exists(input_folder):
        raise ValueError(f"Input folder does not exist: {input_folder}")
    
    files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    
    # 檢查有沒有足夠的圖片進行對比
    if len(files) < 2:
        raise ValueError(f"Need at least 2 images for ECC fitting. Found {len(files)} images in {input_folder}")
    
    results = []
    MIN_AREA = 500

    def calculate_area(binary_img):
        if binary_img is None or binary_img.size == 0:
            return 0
        return int(cv2.countNonZero(binary_img))

    for i in range(len(files)-1):
        try:
            if progress_callback:
                progress_callback(((i+1)/(len(files)-1))*100, f"Processing pair {i+1}/{len(files)-1}")
                
            p1 = os.path.join(input_folder, files[i])
            p2 = os.path.join(input_folder, files[i+1])
            
            # 文件存在性檢查
            if not os.path.exists(p1) or not os.path.exists(p2):
                print(f"File not found: {p1} or {p2}")
                continue
                
            img1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)
            
            if img1 is None or img2 is None:
                print(f"無法載入圖像: {files[i]} 或 {files[i+1]}")
                continue

            # 確保圖片尺寸正確
            if img1.shape != (a4_height, a4_width) or img2.shape != (a4_height, a4_width):
                img1 = cv2.resize(img1, (a4_width, a4_height), interpolation=cv2.INTER_LINEAR)
                img2 = cv2.resize(img2, (a4_width, a4_height), interpolation=cv2.INTER_LINEAR)

            # 驗證A4尺寸
            img1 = validate_a4_size(img1, a4_width, a4_height)
            img2 = validate_a4_size(img2, a4_width, a4_height)

            # 二極化
            _, b1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, b2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # ECC配准
            warp = np.eye(2, 3, dtype=np.float32)
            try:
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-6)
                _, warp = cv2.findTransformECC(
                    b1.astype(np.float32), b2.astype(np.float32), 
                    warp, cv2.MOTION_AFFINE, criteria, None, 3
                )
                b2a = cv2.warpAffine(b2, warp, (b1.shape[1], b1.shape[0]))
            except cv2.error as e:
                print(f"ECC對齊失敗於 {files[i]} -> {files[i+1]}: {str(e)}")
                b2a = b2.copy()

            # 計算面積
            a1 = calculate_area(b1)
            a2a = calculate_area(b2a)
            
            # 確保二值圖像格式正確
            growth_area = cv2.bitwise_and(b2a, cv2.bitwise_not(b1))
            grow = calculate_area(growth_area)
            decomposition_area = cv2.bitwise_and(b1, cv2.bitwise_not(b2a))
            decomp = calculate_area(decomposition_area)

            # 連通組件過濾
            def filter_components(bin_img, min_area):
                if bin_img is None or bin_img.size == 0:
                    return np.zeros((a4_height, a4_width), dtype=np.uint8)
                    
                # 確保輸入是uint8格式
                if bin_img.dtype != np.uint8:
                    bin_img = bin_img.astype(np.uint8)
                    
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
                output = np.zeros_like(bin_img, dtype=np.uint8)
                
                for j in range(1, num_labels):
                    area = int(stats[j, cv2.CC_STAT_AREA])
                    if area < min_area:
                        continue
                    mask = np.where(labels == j, 255, 0).astype(np.uint8)
                    output = cv2.bitwise_or(output, mask)
                return output

            gbin = filter_components(growth_area, MIN_AREA)
            dbin = filter_components(decomposition_area, MIN_AREA)

            # 創建視覺化圖像
            vis = np.zeros((b1.shape[0], b1.shape[1], 3), dtype=np.uint8)
            vis[:, :, 1] = gbin  # 綠色通道 - 生長
            vis[:, :, 2] = dbin  # 紅色通道 - 分解
            
            vis_filename = f"visual_{i+1:02d}.jpg"
            vis_path = os.path.join(output_visual_folder, vis_filename)
            cv2.imwrite(vis_path, vis)

            results.append({
                'Image1': files[i],
                'Image2': files[i+1],
                'Area1': safe_int(a1),
                'Area2_Aligned': safe_int(a2a),
                'Growth_Area': safe_int(grow),
                'Decomposition_Area': safe_int(decomp),
                'Growth_Filtered': safe_int(calculate_area(gbin)),
                'Decomp_Filtered': safe_int(calculate_area(dbin)),
                'Visual': vis_filename
            })

            if progress_callback:
                progress_callback(((i+1)/(len(files)-1))*100, f"Completed pair {i+1}/{len(files)-1}")
                
        except Exception as e:
            print(f"處理圖像對 {files[i]} 和 {files[i+1]} 時出錯: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 即使出錯也要更新進度，這就是人生
            if progress_callback:
                progress_callback(((i+1)/(len(files)-1))*100, f"錯誤: {str(e)[:50]}")
            continue

    # 有結果才存CSV，沒有結果，就絕對不會存
    if results:
        try:
            df = pd.DataFrame(results)
            df.to_csv(output_csv, index=False, encoding='utf-8-sig')
            print(f"ECC results saved to: {output_csv}")
            print(f"Processed {len(results)} image pairs successfully")
        except Exception as e:
            print(f"Error saving CSV: {e}")
            raise
    else:
        raise ValueError("沒有產生任何結果，如同人生 - 請檢查圖像格式和內容")

# 驗證validate_a4_size函数中的問題
def validate_a4_size(image, expected_width, expected_height, tolerance=5):
    """驗證圖像是否為正確的 A4 尺寸"""
    # 修复8: 处理灰度图像
    if len(image.shape) == 2:  # 灰度圖像
        actual_height, actual_width = image.shape
    else:  # 彩色圖像
        actual_height, actual_width = image.shape[:2]
    
    width_diff = abs(actual_width - expected_width)
    height_diff = abs(actual_height - expected_height)
    
    if width_diff > tolerance or height_diff > tolerance:
        print(f"Warning: Image size mismatch - Expected: {expected_width}x{expected_height}, "
              f"Actual: {actual_width}x{actual_height}")
        
        # 強制調整到正確尺寸
        if actual_width != expected_width or actual_height != expected_height:
            resized = cv2.resize(image, (expected_width, expected_height), interpolation=cv2.INTER_LINEAR)
            print(f"Forced resize to A4 dimensions: {expected_width}x{expected_height}")
            return resized
    
    return image

def process_hsv_postprocessing(input_folder, progress_callback=None):
    """HSV 後處理移除小噪點"""
    post_processed_images = os.path.join(input_folder, "post_processed")
    mask_folder = os.path.join(post_processed_images, 'masks')
    image_folder = os.path.join(post_processed_images, 'images')
    os.makedirs(mask_folder, exist_ok=True)
    os.makedirs(image_folder, exist_ok=True)

    lower_red1, upper_red1 = np.array([0, 100, 50]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 100, 50]), np.array([180, 255, 255])
    lower_green, upper_green = np.array([35, 100, 50]), np.array([85, 255, 255])

    min_size = 1874
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', 'jpeg', '.png'))]
    total_files = len(files)
    if total_files == 0:
        raise ValueError("No image files found in the ECC visuals folder")

    for i, fn in enumerate(files):
        img_path = os.path.join(input_folder, fn)
        img = cv2.imread(img_path)
        if img is None:
            continue

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, lower_red1, upper_red1)
        m2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(m1, m2)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        red_large = remove_small_objects_opencv(mask_red, min_size)
        green_large = remove_small_objects_opencv(mask_green, min_size)

        red_small_mask = ((mask_red > 0) & ~(red_large > 0)).astype(np.uint8) * 255
        green_small_mask = ((mask_green > 0) & ~(green_large > 0)).astype(np.uint8) * 255

        mask_small = cv2.bitwise_or(red_small_mask, green_small_mask)
        kernel = np.ones((9, 9), np.uint8)
        mask_small_dil = cv2.dilate(mask_small, kernel, iterations=2)

        mask_save_path = os.path.join(mask_folder, fn.rsplit('.', 1)[0] + '_mask.png')
        cv2.imwrite(mask_save_path, mask_small_dil)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inpainted = cv2.inpaint(img_rgb, mask_small_dil, inpaintRadius=15, flags=cv2.INPAINT_TELEA)
        out_bgr = cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(image_folder, fn)
        cv2.imwrite(out_path, out_bgr)

        if progress_callback:
            progress_callback(((i + 1) / total_files) * 100)

def remove_small_objects_opencv(binary_image, min_size):
    """
    使用純 OpenCV 移除小連通組件
    """
    if binary_image.dtype != np.uint8:
        binary_image = binary_image.astype(np.uint8)

    if binary_image.size == 0 or np.all(binary_image == 0):
        return np.zeros_like(binary_image)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_image, connectivity=8
    )
    output = np.zeros_like(binary_image)

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_size:
            continue
        mask = np.where(labels == i, 255, 0).astype(np.uint8)
        output = cv2.bitwise_or(output, mask)

    return output

def process_images(input_folder, output_folder, a4_width, a4_height, progress_callback=None):
    """嘗試所有方法批量對齊（保持向後兼容）"""
    os.makedirs(output_folder, exist_ok=True)
    files = sorted([f for f in os.listdir(input_folder)
                    if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))])
    if not files:
        raise ValueError("No image files found in folder")

    results_log = []
    for i, fname in enumerate(files, start=1):
        img = cv2.imread(os.path.join(input_folder, fname))
        if img is None:
            continue

        # 第一張很小的圖片做特殊處理
        if i == 1 and img.size < 100000:
            if progress_callback:
                progress_callback(0, f"檢測到小雞雞文件: {fname}, give it something special")
            temp_img = img.copy()
            lab = cv2.cvtColor(temp_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            temp_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            temp_img = cv2.resize(temp_img, (a4_width, a4_height), interpolation=cv2.INTER_CUBIC)
            img = temp_img

        if progress_callback:
            progress_callback((i/len(files))*50, f"Processing {fname}")

        if i == 1:
            ref = preprocess_first_image(img, a4_width, a4_height)
            out = ref
            best_method = "Reference"
            score = ssim = ncc = 1.0
        else:
            resized = center_crop_to_a4(img, a4_width, a4_height)
            out, best_method, score, ssim, ncc = try_all_alignment_methods(
                ref, resized, a4_width, a4_height, progress_callback)

        output_filename = f"aligned_a4_{i:02d}.png"
        cv2.imwrite(os.path.join(output_folder, output_filename), out)

        results_log.append({
            'Image': fname,
            'Output': output_filename,
            'Best_Method': best_method,
            'Combined_Score': score,
            'SSIM': ssim,
            'NCC': ncc
        })

        if progress_callback:
            progress_callback((i/len(files))*100,
                            f"Saved {output_filename} (Method: {best_method}, Score: {score:.4f})")

    results_df = pd.DataFrame(results_log)
    results_df.to_csv(os.path.join(output_folder, "alignment_results.csv"), index=False)
    return results_df

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🌳 Root Turnover Sensor 根系回転センサー ver. 0.44")
        self.geometry("800x800")
        self.configure(bg="#f7f7f7")
        self.folder_path = ""
        self.output_folder = ""
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Status: Ready")
        self.elapsed_time_var = tk.StringVar(value="Elapsed Time: 0.00 s")
        self.queue = queue.Queue()
        self.create_widgets()
        self.after(100, self.process_queue)

    def start_fitting_thread(self):
        """Start ECC fitting thread from button"""
        if not self.folder_path:
            messagebox.showerror("Error", "Please select a folder.")
            return
        try:
            w = int(self.width_entry.get())
            h = int(self.height_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid width and height values.")
            return
        threading.Thread(target=self.start_fitting_process, args=(w, h), daemon=True).start()

    def start_fitting_process(self, a4_width, a4_height):
        """operate ECC fitting"""
        start_time = time.time()
        try:
            csv_path = os.path.join(self.folder_path, "ECC_results.csv")
            vis_folder = os.path.join(self.folder_path, "ECC_visuals")

            def progress_callback(progress, status=None):
                self.queue.put(("progress", progress))
                if status:
                    self.queue.put(("status", status))

            process_ecc_fitting(self.output_folder, csv_path, vis_folder, a4_width, a4_height, progress_callback)
            elapsed = time.time() - start_time
            self.queue.put(("done", elapsed))
        except Exception as e:
            self.queue.put(("error", str(e)))

    def start_hsv_thread(self):
        """Start HSV post-processing thread from button"""
        if not self.folder_path:
            messagebox.showerror("Error", "Please select a folder.")
            return
        ecc_visuals_folder = os.path.join(self.folder_path, "ECC_visuals")
        if not os.path.exists(ecc_visuals_folder):
            messagebox.showerror("Error", "ECC_visuals folder not found. Please run ECC Fitting first.")
            return
        threading.Thread(target=self.start_hsv_process, args=(ecc_visuals_folder,), daemon=True).start()

    def process_queue(self):
        """Process GUI update events from the queue"""
        try:
            while True:
                try:
                    evt = self.queue.get_nowait()
                    if evt[0] == "progress":
                        self.progress_var.set(evt[1])
                    elif evt[0] == "status":
                        self.status_var.set(evt[1])
                    elif evt[0] == "done":
                        self.elapsed_time_var.set(f"Elapsed Time: {evt[1]:.2f} s")
                        messagebox.showinfo("Success", "Processing complete! Check alignment_results.csv for statistics.")
                    elif evt[0] == "error":
                        error_msg = f"Error: {evt[1]}\n\nPlease check:\n1. All images same size\n"
                        error_msg += "2. Correct format (PNG, JPG)\n3. Sufficient features for alignment\n4. Correct A4 dimensions\n"
                        messagebox.showerror("Processing Error", error_msg)
                except queue.Empty:
                    break
        except Exception as e:
            messagebox.showerror("Queue Error", f"Error processing queue: {str(e)}")
        self.after(100, self.process_queue)

    def create_widgets(self):
        # 標題
        tk.Label(self, text="Turnover Sensor v0.43", font=("Helvetica",18,"bold"),
                 fg="#333", bg="#f7f7f7").pack(pady=10)

        # 新功能說明
        info_frame = tk.LabelFrame(self, text="🆕 New Feature 新機能",
                                  bg="#f7f7f7", fg="#0066cc", padx=10, pady=5)
        info_frame.pack(pady=5, fill="x", padx=20)
        tk.Label(info_frame,
                text="Select alignment method or use Auto mode for best results\n整列方法を選択するか、自動モードで最良の結果を得る",
                bg="#f7f7f7", fg="#0066cc", font=("Arial", 9)).pack()

        # Fiji 路徑設置
        fiji_frame = tk.Frame(self, bg="#f7f7f7")
        fiji_frame.pack(pady=5)
        ttk.Button(fiji_frame, text="🔧 Set Fiji Path", command=set_fiji_path).pack(side=tk.LEFT, padx=5)
        tk.Label(fiji_frame, text="Optional: For advanced non-rigid registration",
                 bg="#f7f7f7", fg="#666").pack(side=tk.LEFT)

        # 尺寸設置
        size_frame = tk.Frame(self, bg="#f7f7f7")
        size_frame.pack(pady=10)
        tk.Label(size_frame, text="Width 幅 (px):", bg="#f7f7f7").grid(row=0, column=0, padx=5)
        self.width_entry = tk.Entry(size_frame, width=10)
        self.width_entry.grid(row=0, column=1, padx=5)
        self.width_entry.insert(0, "4961")
        tk.Label(size_frame, text="Height 高さ (px):", bg="#f7f7f7").grid(row=0, column=2, padx=5)
        self.height_entry = tk.Entry(size_frame, width=10)
        self.height_entry.grid(row=0, column=3, padx=5)
        self.height_entry.insert(0, "7016")

        # 方法選擇
        method_frame = tk.LabelFrame(self, text="🎯 Alignment Method Selection 整列方法選択",
                                   bg="#f7f7f7", fg="#0066cc", padx=10, pady=10)
        method_frame.pack(pady=10, fill="x", padx=20)
        self.method_var = tk.StringVar(value="Auto")
        methods = [
            ("Auto (Battle Royale バトル・ロワイアル)", "Auto"),
            ("Euclidean (平移+回転)", "Euclidean"),
            ("Affine (平移+回転+拡縮+せん断)", "Affine"),
            ("OpticalFlow (密な非剛性変形)", "OpticalFlow"),
            ("Homography (透視変換)", "Homography"),
            ("bUnwarpJ (非剛性変形)", "bUnwarpJ")
        ]
        for i, (text, value) in enumerate(methods):
            ttk.Radiobutton(method_frame, text=text, variable=self.method_var,
                           value=value).grid(row=i//2, column=i%2, sticky="w", padx=10, pady=2)

        # 方法說明
        method_info_frame = tk.LabelFrame(self, text="Method Details 方法詳細",
                                         bg="#f7f7f7", padx=10, pady=10)
        method_info_frame.pack(pady=10, fill="x", padx=20)
        methods_text = (
            "• Auto: Automatically selects best method based on quality scores\n"
            "• Euclidean: Translation + Rotation 平移+回転\n"
            "• Affine: Translation + Rotation + Scale + Shear 平移+回転+拡縮+せん断\n"
            "• OpticalFlow: Dense non-rigid deformation 密な非剛性変形\n"
            "• Homography: Perspective transformation 透視変換\n"
            "• bUnwarpJ: Advanced non-rigid registration using Fiji"
        )
        tk.Label(method_info_frame, text=methods_text, bg="#f7f7f7",
                justify=tk.LEFT, font=("Arial", 9)).pack(anchor="w")

        # 操作按钮
        btn_frame = tk.Frame(self, bg="#f7f7f7")
        btn_frame.pack(pady=15)
        ttk.Button(btn_frame, text="📁 Select Folder フォルダを選択",
                  command=self.select_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="🎯 Align Images 整列",
                  command=self.start_align_thread).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="📊 ECC Fitting 擬合分析",
                  command=self.start_fitting_thread).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="🎨 HSV Post-processing HSV後処理",
                  command=self.start_hsv_thread).pack(side=tk.LEFT, padx=5)

        # 状态显示
        tk.Label(self, textvariable=self.status_var, font=("Arial",10),
                fg="#444", bg="#f7f7f7").pack(pady=5)
        ttk.Progressbar(self, variable=self.progress_var, maximum=100).pack(
            pady=15, fill="x", padx=20)
        tk.Label(self, textvariable=self.elapsed_time_var, font=("Arial",12),
                fg="#555", bg="#f7f7f7").pack(pady=10)

        # 版权信息
        tk.Label(self,
                text="Developed by the Forest Utilisation Lab in collaboration with ARATA meeting members. Copyright © 2025.",
                bd=1, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)

    def select_folder(self):
        """select folder"""
        self.folder_path = filedialog.askdirectory(title="Select Folder")
        if self.folder_path:
            self.output_folder = os.path.join(self.folder_path, "aligned_images")
            os.makedirs(self.output_folder, exist_ok=True)
            self.status_var.set(f"Selected folder: {os.path.basename(self.folder_path)}")

    def start_align_thread(self):
        """start to align"""
        if not self.folder_path:
            messagebox.showerror("Error", "Please select a folder.")
            return
        try:
            w = int(self.width_entry.get())
            h = int(self.height_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid width and height values.")
            return
        threading.Thread(target=self.start_align_process, args=(w, h), daemon=True).start()

    def start_align_process(self, a4_width, a4_height):
        """to operate alignment"""
        start_time = time.time()
        try:
            def progress_callback(progress, status=None):
                # This callback will safely forward progress and status to the GUI
                self.queue.put(("progress", progress))
                if status:
                    self.queue.put(("status", status))

            selected_method = self.method_var.get()
            results_df = process_images_with_method(
                self.folder_path,
                self.output_folder,
                a4_width,
                a4_height,
                selected_method,
                progress_callback  # Passed into alignment logic
            )
    
            elapsed = time.time() - start_time
            if selected_method == "Auto":
                method_counts = results_df['Used_Method'].value_counts()
                stats_msg = "Auto mode - Methods used: " + \
                            ", ".join([f"{k}:{v}" for k, v in method_counts.items()])
            else:
                avg_score = results_df['Combined_Score'].mean()
                stats_msg = f"Fixed method: {selected_method}, Avg Score: {avg_score:.4f}"

            self.queue.put(("done", elapsed))
            self.queue.put(("status", f"Complete! {stats_msg}"))

        except Exception as e:
            self.queue.put(("error", str(e)))

    def start_hsv_process(self, ecc_visuals_folder):
        """operate HSV post-processing"""
        start_time = time.time()
        try:
            def progress_callback(progress, status=None):
                self.queue.put(("progress", progress))
                if status:
                    self.queue.put(("status", status))

            process_hsv_postprocessing(ecc_visuals_folder, progress_callback)
            elapsed = time.time() - start_time
            self.queue.put(("done", elapsed))
            self.queue.put(("status", "HSV post-processing completed! Check post_processed folder."))
        except Exception as e:
            self.queue.put(("error", str(e)))


if __name__ == "__main__":
    app = Application()
    app.mainloop()