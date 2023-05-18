# keras vggface model
import tensorflow as tf
from keras.layers import Flatten, Dense, Input, Dropout, Activation, BatchNormalization

from keras_vggface.vggface import VGGFace
from keras.models import Model
# example of loading an image with the Keras API
# since 2021 tensorflow updated the package and moved model directory
from tensorflow.keras.preprocessing import image
import keras_vggface.utils as utils

# image manipulation
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2

# face alignment
from mtcnn.mtcnn import MTCNN

# model metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

# common packages
import os
import numpy as np
import pandas as pd
import pickle

import shutil

# Operations regarding to folder/file
def copy_images(file_paths, source_folder, destination_folder):
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        source_file = os.path.join(source_folder, file_name)
        destination_file = os.path.join(destination_folder, file_name)
        shutil.copyfile(source_file, destination_file)

def get_file_names(folder_path):
    file_names = []
    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_names.append(file_name)
    return file_names

# Easy-to-use Performance metrics
def rmse(x,y):
    return np.sqrt(mean_squared_error(x,y))

def mae(x,y):
    return mean_absolute_error(x,y)

def auc(label, pred):
    return roc_auc_score(label, pred)


# Previous codes for image2array processing; still adopted for single imgae prediction
def imgs_to_array(img_paths, version=1):
    ''' extract features from all images and convert to multi-dimensional array
    Takes:
        img_path: str
        version: int
    Returns:
        np.array
    '''
    imgs = []
    for img_path in img_paths: # += is equivalent to extend @http://noahsnail.com/2020/06/17/2020-06-17-python%E4%B8%ADlist%E7%9A%84append,%20extend%E5%8C%BA%E5%88%AB/
        imgs += [img_to_array(img_path, version)]
    return np.concatenate(imgs)

def process_array(arr, version):
    '''array processing (resize)
    Takes: arr: np.array
    Returns: np.array
    '''
    img = cv2.resize(arr, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = utils.preprocess_input(img, version=version)
    return img

def img_to_array(img_path, version):
    '''conver a SINGLE image to array
    Takes: img_path: str
    Returns: np.array
    '''
    if not os.path.exists(img_path):
        return None  

    img = image.load_img(img_path)
    img = image.img_to_array(img)
    img = process_array(img, version)
    return img

def crop_img(img,x,y,w,h):
    '''crop image
    Takes: img: np.array
           x,y,w,h: int
    Returns: np.array
    '''
    return img[y:y+h,x:x+w,:]

def img_data_generator(data, bs, img_dir, train_mode=True, version = 1): #replace function name later
    """data input pipeline
    Takes:
        data: pd.DataFrame
        bs: batch size
        img_dir: str, directory to the images
        train_mode: bool, if False, take samples from test set to aoivd overfitting
        version: int, keras_vggface version
    Returns:
        features: tuple of (x,y): features and targets
    """
    loop = True
     
    while loop:
        if train_mode:
            x = imgs_to_array(data['path'], version)
            y = data['bmi'].values
            features = (x,y)
        else:
            if len(data) >= bs:
                sampled = data.iloc[:bs,:]
                data = data.iloc[bs:,:]
                features = imgs_to_array(sampled['index'],img_dir, version)
            else: 
                loop = False
        yield features

# Build a prediction class

class FacePrediction(object):

    def __init__(self, img_dir, model_type='vgg16'):
        self.model_type = model_type
        self.img_dir = img_dir
        self.detector = MTCNN()
        if model_type in ['vgg16', 'vgg16_fc6']: # we might use other models, but in that case we need to just version input
            self.version = 1
        else:
            self.version = 2

    def define_model(self, hidden_dim = 64, drop_rate=0.0, freeze_backbone = True): # replace function name later
        ''' initialize the vgg model
        Reference:
            @https://zhuanlan.zhihu.com/p/53116610
            @https://zhuanlan.zhihu.com/p/26934085
        '''
        if self.model_type == 'vgg16_fc6':
            vgg_model = VGGFace(model = 'vgg16', include_top=True, input_shape=(224, 224, 3))
            last_layer = vgg_model.get_layer('fc6').output
            flatten = Activation('relu')(last_layer)
        else:
            vgg_model = VGGFace(model = self.model_type, include_top=False, input_shape=(224, 224, 3))
            last_layer = vgg_model.output
            flatten = Flatten()(last_layer)
        
        if freeze_backbone: # free the vgg layers to fine-tune
            for layer in vgg_model.layers:
                layer.trainable = False
                
        def model_init(flatten, name):
            x = Dense(64, name=name + '_fc1')(flatten)
            x = BatchNormalization(name = name + '_bn1')(x)
            x = Activation('relu', name = name+'_act1')(x)
            x = Dropout(0.2)(x)
            x = Dense(64, name=name + '_fc2')(x)
            x = BatchNormalization(name = name + '_bn2')(x)
            x = Activation('relu', name = name+'_act2')(x)
            x = Dropout(drop_rate)(x)
            x = flatten
            return x
        
        x = model_init(flatten, name = 'bmi')
        bmi_pred = Dense(1, activation='linear', name='bmi')(x) #{'relu': , 'linear': terrible}

        custom_vgg_model = Model(vgg_model.input, bmi_pred)
        custom_vgg_model.compile('adam', 
                                 {'bmi':'mae'}, #{'bmi':'mae'},
                                 loss_weights={'bmi': 1})

        self.model = custom_vgg_model

    def train(self, train_gen, val_gen, train_step, val_step, bs, epochs, callbacks):
        ''' train the model
        Takes: 
            train_data: dataframe
            val_data: dataframe
            bs: int, batch size
            epochs: int, number of epochs
            callbacks: list, callbacks
        Recall the input for img_data_generator: data, bs, img_dir, train_mode=True, version = 1
        '''
        self.model.fit_generator(train_gen, train_step, epochs=epochs,
                                 validation_data=val_gen, validation_steps=val_step,
                                 callbacks=callbacks)


    def evaluate_perf(self, val_data):
        img_paths = val_data['path'].values
        arr = imgs_to_array(img_paths, self.version)
        bmi = self.model.predict(arr)
        metrics = {'bmi_mae':mae(bmi[:,0], val_data.bmi.values)}
        return metrics
    
    def detect_faces(self, img_path, confidence):
        img = image.load_img(img_path)
        img = image.img_to_array(img)
        box = self.detector.detect_faces(img)
        box = [i for i in box if i['confidence'] > confidence]
        res = [crop_img(img, *i['box']) for i in box]
        res = [process_array(i, self.version) for i in res]
        return box, res

    def predict(self, img_input_dir, input_generator=None, input_df=None, show_img=False):
        if os.path.isdir(img_input_dir) and input_generator is not None:
            # Predict using the data generator
              preds = self.model.predict_generator(input_generator)

              if show_img and (input_df is not None):
                  bmi = preds
                  num_plots = len(input_df['path'])
                  ncols = 5
                  nrows = int((num_plots - 0.1) // ncols + 1)
                  fig, axs = plt.subplots(nrows, ncols)
                  fig.set_size_inches(3 * ncols, 3 * nrows)
                  for i, img_path in enumerate(input_df['path']):
                      col = i % ncols
                      row = i // ncols
                      img = plt.imread(img_path)
                      axs[row, col].imshow(img)
                      axs[row, col].axis('off')
                      axs[row, col].set_title('BMI: {:3.1f}'.format(bmi[i, 0], fontsize=10))
              return preds

        else:
            img_path = img_input_dir
            arr = img_to_array(img_path, self.version)
            preds = self.model.predict(arr)
            return preds


    def predict_df(self, img_dir):
        assert os.path.isdir(img_dir), 'input must be directory'
        fnames = os.listdir(img_dir)
        bmi = self.predict(img_dir)
        results = pd.DataFrame({'img':fnames, 'bmi':bmi[:,0]})
        return results
    
    def save_weights(self, model_dir):
        self.model.save_weights(model_dir)

    def load_weights(self, model_dir):
        self.model.load_weights(model_dir)

    def load_model(self, model_dir):
        self.model.load_model(model_dir)

    def predict_faces(self, img_path, show_img = True, color = "white", fontsize = 12, 
                      confidence = 0.95, fig_size = (16,12)):

        assert os.path.isfile(img_path), 'only single image is supported'
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        boxes, faces = self.detect_faces(img_path, confidence)
        preds = [self.model.predict(face) for face in faces]

        if show_img:
            # Create figure and axes
            num_box = len(boxes)
            fig,ax = plt.subplots()
            fig.set_size_inches(fig_size)
            # Display the image
            ax.imshow(img)
            ax.axis('off')
            # Create a Rectangle patch
            for idx, box in enumerate(boxes):
                bmi = preds[idx]
                box_x, box_y, box_w, box_h = box['box']
                rect = patches.Rectangle((box_x, box_y), box_w, box_h, linewidth=1,edgecolor='yellow',facecolor='none')
                ax.add_patch(rect)
                ax.text(box_x, box_y, 
                        'BMI:{:3.1f}'.format(bmi[0,0]),
                       color = color, fontsize = fontsize)
            plt.show()

        return preds