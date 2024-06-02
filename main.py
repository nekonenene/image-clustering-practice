# 必要なライブラリをインポート
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.efficientnet import EfficientNetB2, preprocess_input
from keras.models import Model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# EfficientNetB2モデルをロードし、最終出力層の1つ前の層を出力とするように修正
model = EfficientNetB2(weights='imagenet')
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

# 特徴量を抽出する関数
def extract_features(filepath, model):
    # 画像を260x260ピクセルの配列として読み込みます
    img = load_img(filepath, target_size=(260, 260))

    # 'PIL.Image.Image'型からNumPy配列に変換します
    img = np.array(img)

    # モデルの入力のためにデータの形状を変更します
    # 形状を (1, 260, 260, 3) にします (サンプル数、高さ、幅、チャンネル)
    reshaped_img = img.reshape(1, 260, 260, 3)

    # モデルの入力に合わせて画像を前処理します
    imgx = preprocess_input(reshaped_img)

    # 特徴量ベクトルを取得します
    features = model.predict(imgx)

    # (1, 1280) の形状のデータを (1280,) に変換します
    return features.flatten()

# クラスタ数を設定
CLUSTERS_COUNT = 30
working_path = r"."
imageset_path = r"imageset"

# 作業ディレクトリを変更
os.chdir(working_path)

# 画像ファイル名のリストを保持
image_filenames = []

# 画像ファイルをスキャンし、画像ファイル名をリストに追加
with os.scandir(imageset_path) as files:
    for file in files:
        if file.name.endswith('.jpg'):
            image_filenames.append(file.name)

print(f"Number of image_filenames: {len(image_filenames)}")

# 特徴量の計算と保存
data = {}
p = r"vectors_result.csv"

for image_filename in image_filenames:
    feat = extract_features(imageset_path + '/' + image_filename, model)
    data[image_filename] = feat

data_df = pd.DataFrame.from_dict(data, orient='index')
data_df.to_csv(p)

# 次元削減とクラスタリング
pca = PCA(n_components=2)
x = pca.fit_transform(data_df)

kmeans = KMeans(n_clusters=CLUSTERS_COUNT, random_state=22)
kmeans.fit(x)

groups = {}
for file, cluster in zip(data_df.index, kmeans.labels_):
    if cluster not in groups:
        groups[cluster] = []
    groups[cluster].append(file)

with open('clusters.csv', 'w') as f:
    f.write("cluster, filename\n")
    sorted_groups = sorted(groups.items())
    for cluster, filenames in sorted_groups:
        sorted_filenames = sorted(filenames, key=lambda s: [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)])
        for filename in sorted_filenames:
            f.write(f"{cluster}, {filename}\n")

# クラスタの表示
def view_cluster(cluster):
    plt.figure(figsize=(25, 25))
    files = groups[cluster]
    if len(files) > 30:
        files = files[:29]
    for index, file in enumerate(files):
        plt.subplot(10, 10, index + 1)
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')

# クラスタ数の最適化
sse = []
list_k = list(range(3, 50))

for k in list_k:
    km = KMeans(n_clusters=k, random_state=22)
    km.fit(x)
    sse.append(km.inertia_)

plt.figure(figsize=(6, 6))
plt.plot(list_k, sse)
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance')
plt.savefig("plot.png")
