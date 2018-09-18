import numpy as np
from functools import partial
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt
import urllib.request
import os
import zipfile

def main():
    # Step 1 - Download google's pre-trained neural network
    url = 'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'
    data_dir = '../data/'
    model_name = os.path.split(url)[-1]
    local_zip_file = os.path.join(data_dir, model_name)

    if not os.path.exists(local_zip_file):
        # Download
        model_url = urllib.request.urlopen(url)
        with open(local_zip_file, 'wb') as output:
            output.write(model_url.read())
        
        #Extract
        with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

    # Start with gray image and noise
    img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0
    model_fn = 'tensorflow_inception_graph.pb'

    # Step 2 - Creating Tensorflow Session and Loading Model
    graph = tf.Graph()
    sess = tf.InteractiveSession(graph = graph)

    with tf.gfile.FastGFile(os.path.join(data_dir, model_fn), 'rb') as file:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(file.read())

    t_input = tf.placeholder(np.float32, name='input') # define input tensor
    imagenet_mean = 117.0
    t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
    tf.import_graph_def(graph_def, {'input':t_preprocessed})

    layers = [op.name for op in grapth.get_operations() if op.type == 'Conv2D' and 'import/' in op.name]
    features_num = [int(grapth.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]

    print('Number of layers', len(layers))
    print('Total number of feature channels:', sum(features_num))

if __name__ == '__main__':
    main()