from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from PIL import Image
import cv2
from skimage.feature import hog
import base64
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

def _hash(projection, vec):
    hash_value = (np.dot(projection, vec) > 0).astype('int')
    return "".join(hash_value.astype('str'))

def insert(vecs, num_tables, hash_size, dimensions):
    hash_tables = [{} for _ in range(num_tables)]
    projections = np.random.randn(num_tables, hash_size, dimensions)
    for i, vec in enumerate(vecs):
        for j, table in enumerate(hash_tables):
            hash_value = _hash(projections[j], vec)
            if hash_value in table:
                table[hash_value].append(i)
            else:
                table[hash_value] = [i]
    return hash_tables, projections

def query(vec, hash_tables, projections):
    results = []
    for i, table in enumerate(hash_tables):
        hash_value = _hash(projections[i], vec)
        if hash_value in table:
            results.extend(table[hash_value])
    return results

img_dir = 'Dataset'

# Load image features if they exist, otherwise pre-compute them
if os.path.exists('hog_features.npy'):
    hog_features = np.load('hog_features.npy')
else:
    print('Pre-computing HOG features...')
    images = []
    img_files = os.listdir(img_dir)
    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        img = Image.open(img_path)
        img_array = np.array(img).astype(np.float32) / 255.0
        images.append(img_array)
    images = np.array(images)

    gray_images = []
    for img in images:
        img = img.astype(np.float32)
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_images.append(gray_img)

    hog_features = []
    for gray_img in gray_images:
        features = hog(gray_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
        hog_features.append(features)
    hog_features = np.array(hog_features)

    # Save features for future use
    np.save('hog_features.npy', hog_features)
    print('HOG features pre-computed and saved.')

num_tables = 5
hash_size = 10
dimensions = 324
hash_tables, projections = insert(hog_features, num_tables, hash_size, dimensions)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']

    # Process uploaded image
    img = Image.open(image)
    img_array = np.array(img) / 255.0  
    # Convert image array to unsigned 8-bit integers
    img_array = (img_array * 255).astype(np.uint8)

    # Preprocess and extract features
    gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    query_vec = hog(gray_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')

    # Query LSH for similar images
    results = query(query_vec, hash_tables, projections)
    similar_indices = results[:5]

    # Prepare base64 encoded image data for rendering
    similar_images_data = []
    for i in similar_indices:
        img_path = os.path.join(img_dir, os.listdir(img_dir)[i])
        with open(img_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode('utf-8')
        similar_images_data.append(img_data)

    print("Query Image:", image.filename)  
    
    # Analyze retrieval results based on feature vectors
    query_vector = query_vec.reshape(1, -1)
    print("Retrieval Results:")
    for idx, img_idx in enumerate(similar_indices):
        img_feature = hog_features[img_idx].reshape(1, -1)
        euclidean_dist = euclidean_distances(query_vector, img_feature)[0][0]
        cosine_sim = cosine_similarity(query_vector, img_feature)[0][0]
        print(f"Image {idx + 1}: Euclidean Distance: {euclidean_dist}, Cosine Similarity: {cosine_sim}")

    return render_template('results.html', query_image=image.filename, similar_images_data=similar_images_data)

if __name__ == '__main__':
    app.run(debug=True)
