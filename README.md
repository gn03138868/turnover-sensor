### Turnover Sensor (ver 0.44 stable)
readme.mdの新バージョンを作成中です/
新版本的安裝方法製作中/
New version of readme.md is under building (0.44 added OpticalFlow, Homography, bUnwarpJ, and Auto -Battle Royale. The bUnwarpJ need to install Fiji and pyimageJ. Other methods are the same as usual)

京都大学森林利用学研究室、ARATAミーティング、シティフン ワン  
京都大學森林利用學研究室，ARATA會議，屎地氛 王  
Forest Utilisation lab at Kyoto University, ARATA meeting, & Shitephen Wang  
  
  
##概要/概要/Overview  
Turnover Sensorは、根画像中の成長および分解を解析するためのPythonベースの小ちゃいツールです。このツールは、画像の形状を揃えるために平行移動、回転、せん断、スケーリングを組み合わせた高度な整列手法を提供します（ARATA、ImageJ、Photoshop、またはGIMPのバイナリ画像から…）。また、整列が失敗した場合には簡易的な整列モードに切り替えるフォールバック機能を備えています。さらに、成長および分解計算後に領域接続性の解析とノイズフィルタリングを行います。  
  
Turnover Sensor 是一款基於 Python 的小工具，用於分析影像中的根生長與分解。本工具採用平移、旋轉、剪切及縮放的進階對齊技術，幫助對齊影像中的形狀，並在對齊失敗時切換至簡化模式作為備用方案（可分析來自ARATA、ImageJ、Photoshop或GIMP的黑白圖像...）。此外，工具還提供區域連接分析及在生長和分解計算後的雜訊過濾功能。  
  
The Turnover Sensor is a Python-based little tool designed for analysing root growth and decomposition in images (from binary images of ARATA, ImageJ, Photoshop, or GIMP... ...). This tool employs advanced alignment techniques that combine translation, rotation, shear, and scaling to align shapes within images. If alignment fails, the system automatically switches to a simplified mode as a fallback. Additionally, it includes functionality for analysing region connectivity and filtering noise after growth and decomposition calculations.  
  
  
##特徴/特徴/Features  
三位一体解析：画像整列、成長計算、分解評価を統合。  
高度な整列機能：アフィン変換を用いて正確な画像整列を実現。失敗時は簡易モードに自動切り替え。  
領域接続性解析：最小面積の閾値設定により、接続された領域を特定・処理。  
ノイズフィルタリング：成長と分解分析後のデータ精度を向上。  
A4サイズ標準化：すべての画像をA4サイズ（600 DPI）にリサイズし、パディング処理を実施。  
  
三合一分析：整合影像對齊、生長計算及分解評估。  
進階對齊功能：使用仿射變換實現高精度對齊，失敗時自動切換至簡化模式。  
區域連接分析：可自訂區域最小面積閾值，分析連接的區域。  
雜訊過濾：生長和分解分析後提高數據準確性。  
A4 影像標準化：將影像重設為 A4 尺寸（600 DPI）並自動填充。  
  
Integrated Analysis: Combines image alignment, growth calculation, and decomposition evaluation into a unified workflow.  
Advanced Alignment: Uses affine transformation for precise image alignment, with automatic fallback to a simplified mode in case of failure.  
Region Connectivity Analysis: Identifies and processes connected regions using a configurable minimum area threshold.  
Noise Filtering: Enhances data accuracy by filtering noise after growth and decomposition analysis.  
A4 Standardisation: Resizes all images to A4 dimensions (600 DPI) and applies padding as needed.  
  
  
##必要なソフトウェア/必要軟體/Software Requirements  

1.Pythonバージョン/版本/version  
Python 3.7.6 以上  
  
2.必要なライブラリ/必要函式庫/Required Libraries  
opencv-python（ver. 4.5.5以上）  
numpy（ver. 1.21.6以上）  
pandas（ver. 1.3.5以上）  
tkinter（Python標準インストールに含まれます）  
  
3.追加ソフトウェア/附加軟體/Additional Software  
画像処理：Linuxの場合、libopencv-devをインストールしてください。  
GUIサポート：Tkinterが正しくインストールされている必要があります（ほとんどのPython環境でデフォルトで含まれます）。  
  
  
##インストール方法/安裝方法/Installation Instructions  
(もしすでにARATAの環境が整っている場合は、ARATA環境のルートディレクトリに直接配置してお願いいたします。)  
(如果您已經有ARATA的環境，直接放在ARATA環境的目錄就可以。)   
( If you already have the ARATA environment set up, simply place it in the directory of the ARATA environment.)  
  
1.Pythonのインストール/安裝python/Install python  
公式PythonサイトからPython（3.7.6以上）をダウンロードしてインストールします。  
從官方Python網站下載並安裝 Python（3.7.6以上）  
Download and install Python (version 3.7.6 or higher) from the official Python website.  
https://www.python.org/downloads/  
  
2.ライブラリのインストール/安裝所需函式庫/Install Required Libraries  
必要なライブラリを以下のコマンドでインストールします：  
使用以下指令透過 pip 安裝函式庫：  
  
bash  
pip install opencv-python numpy pandas skimage 
  
  
3.インストール確認/安裝確認/Verify Installation  
ライブラリが正しくインストールされているか確認します：  
確保函式庫已正確安裝：  
  
bash  
python -m pip show opencv-python numpy pandas skimage 
  
  
##使用方法/使用方法/Usage Instructions  
1.このリポジトリをクローンまたはダウンロードします。  
1.將此專案下載或複製至平常執行的資料夾中。  
1.Clone or download this repository to your local system.  
  
2.処理したい画像を含む入力フォルダを準備します。  
2.準備包含待處理影像的輸入資料夾。  
2.Prepare an input folder containing the images you wish to process.  
  
3.以下のコマンドでスクリプトを実行します：  
3.運行以下指令啟動工具：  
3.Run the script with the following command:  
  
bash  
python turnover_sensor_044.py  

3.1. If ModuleNotFoundError: No module named 'imagej' appeared
commend below
pip install pyimagej --user
  
4.GUIの指示に従って、入力フォルダを選択してください。  
4.根據 GUI 指引，選擇輸入資料夾。  
4.Follow the GUI instructions to select input folders.  
  
  
##出力ファイル/輸出檔案/Output Files  
整列画像：A4サイズに標準化されたPNGファイルとして保存されます。  
解析結果：ECCフィッティング解析が有効化されている場合、結果はCSV形式で保存されます。  
可視化：処理済みの可視化結果は指定した出力フォルダに保存されます。  
  
對齊影像：以標準化 A4 尺寸的 PNG 檔案存檔。  
分析結果：若啟用 ECC 配準分析，結果將保存為 CSV 格式。  
可視化輸出：處理後的可視化結果存儲於指定的輸出資料夾。  
  
Aligned Images: Saved as standardised PNG files resized to A4 dimensions.  
Analysis Results: If ECC fitting analysis is enabled, results will be saved in CSV format.  
Visualisations: Processed visualisation outputs will be saved in the specified output folder.  
  
  
##サポート/支援/Support  
ご質問やバグの報告、機能に関するリクエストがございましたら、どうぞお気軽にプロジェクトのメンテナまでご連絡いただけますようお願い申し上げます。  
如有疑問、錯誤回報或功能建議，請聯繫維護人員，感謝您，如果他還活著的話。  
If you have any questions, bug reports, or feature requests, please feel free to contact the maintainer if he is still alive in the world.  
  
##メンテナ/維護人員/Maintainer：  
シティフン ワン/屎地氛/Shitephen Wang  
Eメール/電子信箱/Email：gn03138868@gmail.com  
