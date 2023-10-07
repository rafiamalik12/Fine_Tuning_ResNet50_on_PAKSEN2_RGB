
from __imports__ import *
from __config__ import *

# Load the ResNet50 model
resnet_model = ResNet50(weights=weights, include_top=False, input_shape=input_shape)

# Freeze all layers except last 20 layers
for layer in resnet_model.layers[-20:]:
    layer.trainable = False

# checking the trainable status of the individual layers
for layer in resnet_model.layers:
    print(layer, layer.trainable)

# create model
model = models.Sequential()


# Add the resnet50 convolutional model
model.add(resnet_model)

# Adding new Layers
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))


# Data Generators

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


# reading input filenames and labels
traindf = pd.read_csv(train_csv_path)
valdf = pd.read_csv(val_csv_path)

# loading training data (80%)
training_set = train_datagen.flow_from_dataframe(
        dataframe=traindf,
        directory=input_data_path+'/TrainSet',
        x_col="id",
        y_col="labels",
        batch_size=32,
        seed=42,
        shuffle=True,
        class_mode="categorical",
        target_size=(64, 64))

val_set = val_datagen.flow_from_dataframe(
        dataframe=valdf,
        directory=input_data_path+'/ValSet',
        x_col="id",
        y_col="labels",
        batch_size=32,
        seed=42,
        shuffle=True,
        class_mode="categorical",
        target_size=(64, 64))

# lets determine dataset characteristics
print('Training Data: ', training_set[0][0].shape)
print('Validation Data: ', val_set[0][0].shape)

# now shape of a single image
print('Shape of single image:', training_set[0][0][0].shape)


# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(learning_rate=1e-6),
              metrics=['acc'])


# checkpoint

checkpoint = ModelCheckpoint(os.path.join(model_output_path, model_name),
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             mode='max')

earlystopper = EarlyStopping(monitor='val_acc',
                             patience=5,
                             mode='max',
                             restore_best_weights=True)

callbacks_list = [checkpoint, earlystopper]

# Train the model
history = model.fit(
          training_set, 
          steps_per_epoch=(training_set.samples/32),
          epochs=epochs,
          callbacks=callbacks_list,
          verbose=1,
          validation_data=val_set,
          validation_steps=(val_set.samples/32))


#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(os.path.join(model_output_path, 'train_val_acc.png'))

plt.clf()

# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(os.path.join(model_output_path, 'train_val_loss.png'))

