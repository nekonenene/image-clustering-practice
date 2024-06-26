### From https://towardsdatascience.com/how-to-cluster-images-based-on-visual-similarity-cd6e7209fe34
# for loading/processing the images
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

# models
from keras.applications.vgg16 import VGG16
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle
import shutil

CLUSTERS_COUNT = 30 # Number of clusters of similar images to group together

working_path = r"."
imageset_dir_path = 'imageset'

# change the working directory to the path where the images are located
os.chdir(working_path)

# this list holds all the image filename
image_filenames = []

# creates a ScandirIterator aliased as files
with os.scandir(imageset_dir_path) as files:
    # loops through each file in the directory
    for file in files:
        if file.name.endswith('.jpg'):
            # adds only the image files to the image_filenames list
            image_filenames.append(file.name)

print(f"Number of image_filenames: {len(image_filenames)}")

model = VGG16()
model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

def extract_features(filepath, model):
    # load the image as a 224x224 array
    img = load_img(filepath, target_size=(224, 224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img)
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1, 224, 224, 3)
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx)
    return features

data = {}
p = r"vectors_result.csv"

# lop through each image in the dataset
for image_filename in image_filenames:
    # try to extract the features and update the dictionary
    try:
        feat = extract_features(imageset_dir_path + '/' + image_filename, model)
        data[image_filename] = feat
    # if something fails, save the extracted features as a pickle file (optional)
    except Exception as e:
        print(f"Error: {e}")
        with open(p, 'wb') as file:
            pickle.dump(data, file)

print(f"Number of data.keys: {len(data.keys())}")
print(f"Number of data.values: {len(data.values())}")

# get a list of the filenames
filenames = np.array(list(data.keys()))

# get a list of just the features
feat = np.array(list(data.values()))

# reshape so that there are 210 samples of 4096 vectors
feat = feat.reshape(-1, 4096)

# # get the unique labels (from the image_labels.csv)
# df = pd.read_csv('image_labels.csv')
# label = df['label'].tolist()
# unique_labels = list(set(label))

# print (f"Unique labels: {unique_labels}")

# reduce the amount of dimensions in the feature vector
pca = PCA(n_components=100, random_state=22)
pca.fit(feat)
x = pca.transform(feat)

# cluster feature vectors
kmeans = KMeans(n_clusters=CLUSTERS_COUNT, random_state=22)
kmeans.fit(x)

# holds the cluster id and the images { id: [images] }
groups = {}
for file, cluster in zip(filenames, kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file)
    else:
        groups[cluster].append(file)

# Generate groups CSV each picture order by cluster id and picture name
with open('clusters.csv', 'w') as f:
    f.write("cluster, filename\n") # header
    sorted_groups = sorted(groups.items(), key=lambda x: x[0])
    print(sorted_groups)

    for cluster, filenames in sorted_groups:
        # Sort by filename while keeping numbers order
        sorted_filenames = sorted(filenames, key=lambda s: [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)])
        for filename in sorted_filenames:
            f.write(f"{cluster}, {filename}\n")

# Copy clusters.csv into imageset_dir_path
shutil.copy('clusters.csv', imageset_dir_path)

# function that lets you view a cluster (based on identifier)
def view_cluster(cluster):
    plt.figure(figsize = (25, 25));
    # gets the list of filenames for a cluster
    files = groups[cluster]
    # only allow up to 30 images to be shown at a time
    if len(files) > 30:
        print(f"Clipping cluster size from {len(files)} to 30")
        files = files[:29]
    # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(10, 10, index+1);
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')


# this is just incase you want to see which value for k might be the best
sse = []
list_k = list(range(3, 50))

for k in list_k:
    km = KMeans(n_clusters=k, random_state=22)
    km.fit(x)

    sse.append(km.inertia_)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse)
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance');

plt.savefig("plot.png")
