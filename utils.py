import tensorflow as tf
from tensorflow.contrib import slim
import os
import numpy as np
from glob import glob
import cv2
import pickle

class Image_data:

    def __init__(self, img_height, img_width, channels, dataset_path, augment_flag):
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.augment_flag = augment_flag

        self.dataset_path = dataset_path
        self.image_path = os.path.join(dataset_path, 'images')
        self.text_path = os.path.join(dataset_path, 'text')

        self.embedding_pickle = os.path.join(self.text_path, 'char-CNN-RNN-embeddings.pickle')
        self.image_filename_pickle = os.path.join(self.text_path, 'filenames.pickle')


        self.image_list = []


    def image_processing(self, filename, vector):
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels, dct_method='INTEGER_ACCURATE')
        img = tf.image.resize_images(x_decode, [self.img_height, self.img_width])
        img = tf.cast(img, tf.float32) / 127.5 - 1


        if self.augment_flag :
            augment_height_size = self.img_height + (30 if self.img_height == 256 else int(self.img_height * 0.1))
            augment_width_size = self.img_width + (30 if self.img_width == 256 else int(self.img_width * 0.1))

            img = tf.cond(pred=tf.greater_equal(tf.random_uniform(shape=[], minval=0.0, maxval=1.0), 0.5),
                          true_fn=lambda : augmentation(img, augment_height_size, augment_width_size),
                          false_fn=lambda : img)

        return img, vector

    def preprocess(self):
        with open(self.embedding_pickle, 'rb') as f:

            self.embedding = pickle._Unpickler(f)
            self.embedding.encoding = 'latin1'
            self.embedding = self.embedding.load()
            self.embedding = np.array(self.embedding) # (8855, 10, 1024)

        with open(self.image_filename_pickle, 'rb') as f:
            # ['002.Laysan_Albatross/Laysan_Albatross_0002_1027', '002.Laysan_Albatross/Laysan_Albatross_0003_1033', ... ]

            x_list = pickle.load(f)

            for x in x_list :
                folder_name = x.split('/')[0]
                file_name = x.split('/')[1] + '.jpg'

                self.image_list.append(os.path.join(self.image_path, folder_name, file_name))


def load_test_image(image_path, img_width, img_height, img_channel):

    if img_channel == 1 :
        img = cv2.imread(image_path, flags=cv2.IMREAD_GRAYSCALE)
    else :
        img = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, dsize=(img_width, img_height))

    if img_channel == 1 :
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)
    else :
        img = np.expand_dims(img, axis=0)

    img = img/127.5 - 1

    return img


def preprocessing(x):
    x = x/127.5 - 1 # -1 ~ 1
    return x

def preprocess_fit_train_image(images, height, width):
    images = tf.image.resize(images, size=[height, width], method=tf.image.ResizeMethod.BILINEAR)
    images = adjust_dynamic_range(images)

    return images

def adjust_dynamic_range(images):
    drange_in = [0.0, 255.0]
    drange_out = [-1.0, 1.0]
    scale = (drange_out[1] - drange_out[0]) / (drange_in[1] - drange_in[0])
    bias = drange_out[0] - drange_in[0] * scale
    images = images * scale + bias
    return images

def augmentation(image, augment_height, augment_width):
    seed = np.random.randint(0, 2 ** 31 - 1)

    ori_image_shape = tf.shape(image)
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.resize(image, size=[augment_height, augment_width], method=tf.image.ResizeMethod.BILINEAR)
    image = tf.random_crop(image, ori_image_shape, seed=seed)


    return image

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
    return ((images+1.) / 2) * 255.0

def imsave(images, size, path):
    images = merge(images, size)
    images = cv2.cvtColor(images.astype('uint8'), cv2.COLOR_RGB2BGR)

    return cv2.imwrite(path, images)


def post_process_generator_output(generator_output):

    drange_min, drange_max = -1.0, 1.0
    scale = 255.0 / (drange_max - drange_min)

    scaled_image = generator_output * scale + (0.5 - drange_min * scale)
    scaled_image = np.clip(scaled_image, 0, 255)

    return scaled_image

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img

def return_images(images, size) :
    x = merge(images, size)

    return x

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def str2bool(x):
    return x.lower() in ('true')

def get_one_hot(targets, nb_classes):

    x = np.eye(nb_classes)[targets]

    return x

def pytorch_xavier_weight_factor(gain=0.02, uniform=False) :

    if uniform :
        factor = gain * gain
        mode = 'FAN_AVG'
    else :
        factor = (gain * gain) / 1.3
        mode = 'FAN_AVG'

    return factor, mode, uniform

def pytorch_kaiming_weight_factor(a=0.0, activation_function='leaky_relu', uniform=False) :

    if activation_function == 'relu' :
        gain = np.sqrt(2.0)
    elif activation_function == 'leaky_relu' :
        gain = np.sqrt(2.0 / (1 + a ** 2))
    elif activation_function == 'tanh' :
        gain = 5.0 / 3
    else :
        gain = 1.0

    if uniform :
        factor = gain * gain
        mode = 'FAN_IN'
    else :
        factor = (gain * gain) / 1.3
        mode = 'FAN_IN'

    return factor, mode, uniform
