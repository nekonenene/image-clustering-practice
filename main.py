import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# 特定のフォルダのパスを指定
folder_path = 'imageset'

# MobileNetV3モデルをロードし、特徴抽出用に調整
base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

# 画像の特徴量を抽出する関数
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features.flatten()

# フォルダ内のJPEG画像を読み込む
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg')]
features_list = []
filenames = []

for filename in image_files:
    file_path = os.path.join(folder_path, filename)
    features = extract_features(file_path)
    features_list.append(features)
    filenames.append(filename)

# 特徴量をNumPy配列に変換
features_array = np.array(features_list)

# 特徴量をスケーリング
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_array)

# 適切なepsの値を見つけたらDBSCANを使用してクラスタリング
eps_value = 12 # この値が小さいほどクラスタ数が増える
min_samples_value = 1
dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
labels = dbscan.fit_predict(features_array)

# labels, filenames の組み合わせをCSVとして出力。ラベルの値が小さい順、同じラベルの中ではファイル名の順になるよう並び替えて出力
output_df = pd.DataFrame({'label': labels, 'filename': filenames})
output_df = output_df.sort_values(['label', 'filename'])
output_df.to_csv('clusters.csv', index=False)
