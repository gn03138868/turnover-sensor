It is a prototype of root turnover sensor.
これは根のターンオーバーセンサーのプロトタイプです。
這是一個根部週轉率感測器的原型。

Step 1. operate Tif_JPGE.py to transfer tif images to jpg images
ステップ1. Tif_JPGE.py を実行して、TIF画像をJPG画像に変換します。
步驟1. 運行 Tif_JPGE.py，將 TIF 圖片轉換為 JPG 圖片。

Step 2. operate align_images_ECC.py to align images for further steps
ステップ2. align_images_ECC.py を実行して、画像を位置合わせし、次のステップに備えます。
步驟2. 運行 align_images_ECC.py，對圖片進行校準，以便進行後續步驟。

Step 3. operate ECC_fitting.py to calculate the root increment and decrement by the output of post-process images from ARATA
ステップ3. ECC_fitting.py を実行して、ARATAによる後処理画像を基に根の増加および減少を計算します。
步驟3. 運行 ECC_fitting.py，利用 ARATA 的後處理輸出圖片計算根部的增長與減少量。

(The higher accuracy the output from ARATA, the higher accuracy the turnover of roots represented.)
（ARATAからの出力精度が高いほど、根のターンオーバー精度も高く表現されます。）
（ARATA 的輸出精度越高，根部週轉率的準確性也會越高。）
