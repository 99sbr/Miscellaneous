import os
import shutil
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import applications
from keras import optimizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping
from keras.layers import Dropout, Dense, Activation, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

IM_WIDTH, IM_HEIGHT = 256, 256
TF_EPOCH = 15
FT_EPOCH = 30
BATCH_SIZE = 8
NB_IV3_LAYERS_TO_FREEZE = 172


def add_callback(phase):
    if phase == 'tf':
        weightsOutputFile = 'ModelCheckpoints/InceptionV3_tf.{epoch:02d}-{val_accuracy:.3f}.hdf5'
    elif phase == 'ft':
        weightsOutputFile = 'ModelCheckpoints/InceptionV3_ft.{epoch:02d}-{val_accuracy:.3f}.hdf5'

    tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0,
                               batch_size=32, write_graph=True, write_images=True)
    learning_rate_reduction = ReduceLROnPlateau(
        monitor='val_accuracy', patience=5, verbose=1, cooldown=3, factor=0.5, min_lr=0.00001)

    checkpoint = ModelCheckpoint(weightsOutputFile, monitor='val_accuracy',
                                 verbose=1, save_best_only=True, mode='auto', period=1)
    early = EarlyStopping(monitor='val_accuracy', min_delta=0,
                          patience=10, verbose=1, mode='auto')
    return [checkpoint, learning_rate_reduction]


def setup_to_transfer_learn(model, base_model):
    """
    Freeze all layers and compile the model
    """
    for layer in base_model.layers:
        layer.trainable = False
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.Adam(), metrics=['accuracy'])

    return model


def setup_to_finetune(model):
    """
    Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
    note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
    Args:
      model: keras model
    """
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=1e-4), metrics=['accuracy'])

    return model


def add_new_last_layer(base_model, nb_classes):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024)(x)
    x = Activation('relu')(x)

    x = Dropout(0.3)(x)
    x = Dense(512)(x)
    x = Activation('relu')(x)

    predictions = Dense(nb_classes, activation='softmax')(x)
    # Creating final model
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def plot_training(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.figure(1)
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    plt.savefig('history' + str(1) + '.png')

    # summarize history for loss
    plt.figure(2)
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    plt.savefig('history' + str(2) + '.png')


def build_model(train_df, val_df):
    nb_train_samples = len(train_df)
    nb_validation_samples = len(val_df)
    # Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2
    )

    val_datagen = ImageDataGenerator(
        rescale=1. / 255
    )

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory='./training_data/',
        x_col="image_id",
        y_col="label",
        target_size=(256, 256),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb')

    np.save('class_indices', train_generator.class_indices)

    validation_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory='./validation_data/',
        x_col="image_id",
        y_col="label",
        target_size=(256, 256),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb'
    )

    base_model = applications.InceptionV3(weights="imagenet", include_top=False,
                                          input_shape=(IM_WIDTH, IM_HEIGHT, 3))

    model = add_new_last_layer(base_model, len(train_df.label.value_counts()))
    # transfer learning
    print('----------- Transfer Learning ------------>')
    model = setup_to_transfer_learn(model, base_model)
    # Train model
    history1 = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_generator.n // train_generator.batch_size,
        epochs=TF_EPOCH,
        shuffle=True,
        validation_data=validation_generator,
        validation_steps=validation_generator.n // validation_generator.batch_size,
        callbacks=add_callback('tf')
    )
    # fine-tuning
    print('----------- Fine Tuning ------------>')
    model = setup_to_finetune(model)
    history2 = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_generator.n // train_generator.batch_size,
        epochs=FT_EPOCH,
        shuffle=True,
        validation_data=validation_generator,
        validation_steps=validation_generator.n // validation_generator.batch_size,
        callbacks=add_callback('ft')
    )

    print('Training Completed. Saving plots.')
    plot_training(history1)
    plot_training(history2)


if __name__ == '__main__':
    ##TODO Add configuration file for relative folder and image paths
    # Image Data Dir
    train_img_dir = "a0409a00-8-dataset_dp/train_img"

    # load meta-data
    train_meta = pd.read_csv('./a0409a00-8-dataset_dp/train.csv')
    train_meta['image_id'] = train_meta.image_id.apply(lambda x: x + '.png')
    print('--------Category Stats--------\n')
    print(train_meta.label.value_counts())

    print('Training Data Shape:', train_meta.shape)

    train_files, validate_files, y_train, y_val = train_test_split(
        train_meta['image_id'], train_meta['label'], test_size=0.15, stratify=train_meta['label'], random_state=42)

    train_df = pd.DataFrame({'image_id': train_files, 'label': y_train})
    val_df = pd.DataFrame({'image_id': validate_files, 'label': y_val})

    print('--------Category Stats Train df--------\n')
    print(train_df.label.value_counts())

    print('--------Category Stats Test df--------\n')
    print(val_df.label.value_counts())

    print('Train samples: {}'.format(len(train_files)))
    print('Validation samples: {}'.format(len(validate_files)))

    train_dir = 'training_data'
    val_dir = 'validation_data'

    os.mkdir(train_dir) if not os.path.isdir(train_dir) else None
    os.mkdir(val_dir) if not os.path.isdir(val_dir) else None

    for row in train_df.iterrows():
        shutil.copy(os.path.join(train_img_dir, row[1][0]), train_dir)

    for row in val_df.iterrows():
        shutil.copy(os.path.join(train_img_dir, row[1][0]), val_dir)

    # Model preparation
    build_model(train_df, val_df)
