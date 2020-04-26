import math
import os
import shutil
import sys
import pandas as pd
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing import image


class ImageUtility:
    def __init__(self):
        pass

    def get_file_list(self, data_dir):
        file_list = []
        for root, dirs, files in os.walk(data_dir):
            if bool(files):
                for file in files:
                    file_list.append(os.path.join(root, file))

        sys.stdout.flush()
        print('Total number of files found {}'.format(len(file_list)))
        return file_list

    def get_images(self, file_list):
        target_size = (256, 256)
        images = []
        for file in file_list:
            img = Image.open(file)
            if img.size != target_size:
                img = img.resize(target_size)
            img = image.img_to_array(img)
            # img_tensor = np.expand_dims(img,axis=0)
            img /= 255.
            # img = preprocess_input(img)
            images.append(img)
        return np.array(images)

    def imageLoader(self, files, batch_size):
        L = len(files)
        while True:
            batch_start = 0
            batch_end = batch_size
            while batch_start < L:
                limit = min(batch_end, L)
                X = self.get_images(files[batch_start:limit])
                yield (X)
                batch_start += batch_size
                batch_end += batch_size

    @staticmethod
    def get_class_dict():
        class_indices = np.load('class_indices.npy').item()
        return class_indices

    def get_key(self, val):
        class_indices = ImageUtility().get_class_dict()
        for key, value in class_indices.items():
            if val == value:
                return key
        return "key doesn't exists "

    def move_files(self, input_zip, prediction_dir):
        for filepath, y_class in input_zip:
            if not os.path.isdir(os.path.join(prediction_dir, y_class)):
                os.mkdir(os.path.join(prediction_dir, y_class))
            else:
                pass
            src = filepath
            dst = os.path.join(prediction_dir, y_class)
            shutil.copy(src, dst)


class Prediction:
    def __init__(self, model):
        self.batch_size = 32
        self.model = model
        self.image_utility = ImageUtility()
        self.data_dir = "a0409a00-8-dataset_dp/test_img"
        self.move_data_dir = 'prediction'

    def predict_on_data(self):
        filelist = self.image_utility.get_file_list(self.data_dir)
        predict = self.model.predict_generator(self.image_utility.imageLoader(filelist, self.batch_size),
                                               steps=int(math.ceil(len(filelist) / self.batch_size)), verbose=1)

        predicted_classes = [np.argmax(probability) for probability in predict]
        predicted_actual_class = [self.image_utility.get_key(val) for val in predicted_classes]
        assert len(filelist) == len(predicted_actual_class)
        self.image_utility.move_files(zip(filelist, predicted_actual_class), self.move_data_dir)
        data = pd.DataFrame({'image_id':filelist,'label':predicted_actual_class})
        data.to_csv('test_pred.csv',index=False)


if __name__ == '__main__':
    ##TODO Add configuration
    model = load_model('ModelCheckpoints/InceptionV3_ft.27-0.813.hdf5', compile=False)
    predict = Prediction(model)
    predict.predict_on_data()
