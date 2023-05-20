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
from PIL import Image, ImageDraw
import cv2
import keras_vggface.utils as utils
from tensorflow.keras.layers import Conv2D, MaxPooling2D

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
from tqdm import tqdm
import tempfile
import hashlib

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
    desired_size = (224, 224)
    img = cv2.resize(arr, desired_size)
    img = img * (1./255)
    #img = np.expand_dims(img, axis=0)
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

def array_to_img(arr):
    '''Converts a numpy array to an image.
    Takes: arr: np.array
    Returns: PIL.Image
    '''
    # Convert array to image
    img = Image.fromarray(np.uint8(arr*255))
    return img

        
# build a ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def single_test_generator(img_path, target_size=(224, 224), batch_size=1):
    '''Generate a single test generator from an image file.
    Takes:
        - img_path: str, path to the image file
        - target_size: tuple, target size for image resizing (default: (224, 224))
        - batch_size: int, batch size for the generator (default: 32)
    Returns:
        - single_test_gen: ImageDataGenerator, generated image generator
    '''
    # Load the image
    img = load_img(img_path, target_size=target_size)
    # Convert the image to an array
    img_array = img_to_array(img)
    # Reshape the array to match the expected input shape of the model
    img_array = img_array.reshape((1,) + img_array.shape)
    # Create an instance of ImageDataGenerator
    test_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)
    # Generate the images
    single_test_gen = test_datagen.flow(img_array, batch_size=batch_size)

    return single_test_gen

# Create the ImageDataGenerator for the test_data
test_datagen = ImageDataGenerator(
    #preprocessing_function=lambda x: (x - mean_pixel_value) / 255.0,
    rescale = 1./255)

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
                features = imgs_to_array(sampled['index'], img_dir, version)
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
            #flatten = Activation('relu')(last_layer)
        else:
            vgg_model = VGGFace(model = self.model_type, include_top=False, input_shape=(224, 224, 3))
            last_layer = vgg_model.output
            flatten = Flatten()(last_layer)
        
        if freeze_backbone: # free the vgg layers to fine-tune
            for layer in vgg_model.layers:
                layer.trainable = False
                
    ## extra 1 conv layer
        def model_init(x, name):
            x = Conv2D(filters=32, kernel_size=(3,3), input_shape=(244, 244, 3),
                       activation='relu', name=name+'_con1')(x)
            x = MaxPooling2D(pool_size=(2,2))(x)
            
            # x = Conv2D(filters=64, kernel_size=(3,3), input_shape=(244, 244, 3),
            #            activation='relu', name=name+'_con2')(x)
            # x = MaxPooling2D(pool_size=(2,2))(x)      
            # x = Dense(512, name=name+'_fc1')(x)
            
            x = Flatten()(x)
            
            # x = BatchNormalization(name=name+'_bn2')(x)
            # x = Activation('relu', name=name+'_act1')(x)
            # x = Dropout(drop_rate)(x)
            # x = Dense(256, name=name+'_fc2')(x)
            # x = BatchNormalization(name=name+'_bn3')(x)
            # x = Activation('relu', name=name+'_act2')(x)
            # x = Dropout(drop_rate)(x)
            # x = Dense(128, name=name+'_fc3')(x)
            # x = BatchNormalization(name=name+'_bn4')(x)
            # x = Activation('relu', name=name+'_act3')(x)
            # x = Dropout(drop_rate)(x)
            return x
        
        x = model_init(last_layer, name = 'bmi')
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

    def crop_image_around_face(self, img, box, crop_percentage):
        x, y, width, height = box['box']
        center_x = x + (width // 2)
        center_y = y + (height // 2)
        crop_width = int(width * crop_percentage)
        crop_height = int(height * crop_percentage)
        crop_left = max(0, center_x - (crop_width // 2))
        crop_top = max(0, center_y - (crop_height // 2))
        crop_right = min(img.width, crop_left + crop_width)
        crop_bottom = min(img.height, crop_top + crop_height)
        cropped_img = img.crop((crop_left, crop_top, crop_right, crop_bottom))
        return cropped_img

    def process_input_image(self, img_input_path):
        img = Image.open(img_input_path)

        # Check image size
        if img.size == (244, 244):
            return img_input_path
        else:
            # Detect faces and crop
            confidence_threshold = 0.5
            boxes, cropped_images = self.detect_faces(img_input_path, confidence_threshold)

            if len(cropped_images) > 0:
                # Save the cropped image in a temporary folder
                tmp_folder = 'tmp'
                os.makedirs(tmp_folder, exist_ok=True)

                # Generate hash value from the image input path
                hash_value = hashlib.sha1(img_input_path.encode()).hexdigest()

                tmp_img_path = os.path.join(tmp_folder, hash_value + 'temp_image.bmp')

                # Print confidence for each detected face
                for i, box in enumerate(boxes):
                    confidence = box['confidence']
                    print(f"Face {i + 1}: Confidence - {confidence}")

                    # Crop the image around the detected face
                    cropped_img = self.crop_image_around_face(img, box, crop_percentage=1.25)

                # Save the cropped image
                cropped_img.save(tmp_img_path)

                return tmp_img_path
            else:
                # No faces detected, return the original image
                return img_input_path


    def predict_external(self, img_input_dir, input_df=None, show_img=False):
        if os.path.isdir(img_input_dir) and input_df is not None:
            assert not os.path.isdir(img_input_dir), "Input should be a path, not a directory"

        else:
            # Single image input
            single_test_path = self.process_input_image(img_input_dir)
            single_test_gen = single_test_generator(single_test_path)

            if show_img:
                img_path = img_input_dir
                img = plt.imread(single_test_path)
                fig, ax = plt.subplots()
                ax.imshow(img)
                ax.axis('off')
                preds = self.model.predict_generator(single_test_gen)
                ax.set_title('BMI: {:3.1f}'.format(preds[0, 0], fontsize=10))
                plt.show()

            preds = self.model.predict_generator(single_test_gen)
            return preds


    def predict_external_dir(self, img_input_dir, input_df=None, show_img=False):
        '''
        This function deals with mutiple input, because when input data is affected when importing data with datagen.from_dataframe() compared to .from_flow
        '''
        if os.path.isdir(img_input_dir) and input_df is not None:
            # Crop the images and makde a temporary df & dir
            test_df = input_df
            processed_img_paths = [self.process_input_image(i) for i in test_df['path']]
            processed_img_names = [i.split('/')[-1] for i in processed_img_paths]
            processed_img_dir = '/'.join(processed_img_paths[0].split('/')[:-1])
            test_df['processed_paths'], test_df['processed_names'] = processed_img_paths, processed_img_names

            # Make prediction
            test_set_gen = test_datagen.flow_from_dataframe(
                test_df,
                directory = img_input_dir,
                x_col='name',
                y_col='bmi',
                target_size=(244, 244),
                batch_size=32,
                color_mode='rgb',
                class_mode='raw')

            preds = self.model.predict_generator(test_set_gen)

            if show_img and (test_df is not None):
                bmi = preds
                num_plots = len(test_df['path'])
                ncols = 5
                nrows = int((num_plots - 0.1) // ncols + 1)
                fig, axs = plt.subplots(nrows, ncols)
                fig.set_size_inches(3 * ncols, 3 * nrows)
                for i, img_path in enumerate(test_df['path']):
                    col = i % ncols
                    row = i // ncols
                    img = plt.imread(img_path)
                    axs[row, col].imshow(img)
                    axs[row, col].axis('off')
                    axs[row, col].set_title('BMI: {:3.1f}'.format(bmi[i, 0], fontsize=10))
            return preds


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
            single_test_gen = single_test_generator(img_input_dir)
            
            if show_img:
                img_path = img_input_dir
                img = plt.imread(img_path)
                fig, ax = plt.subplots()
                ax.imshow(img)
                ax.axis('off')
                #preds = self.model.predict(img_to_array(img_path, self.version))
                preds = self.model.predict_generator(single_test_gen)
                ax.set_title('BMI: {:3.1f}'.format(preds[0, 0], fontsize=10))
                plt.show()

            preds = self.model.predict_generator(single_test_gen)
            #preds = self.model.predict(img_to_array(img_path, self.version))
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