from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Flatten, Dense, Input
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import ResNet50
from imutils import paths
import numpy as np
import cv2
import os
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

dataset = "Dataset"
label = "lb.pickle"

LABELS = set(["Fusarium Head Blight", "Healthy Wheat", "Leaf Rust", "Tan Spot", "Unknown"])

imagePaths = list(paths.list_images(dataset))
data = []
labels = []
for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    if label not in LABELS:
        continue

    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    data.append(image)
    labels.append(label)

data = np.array(data)
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.25, stratify=labels, random_state=42)

trainAug = ImageDataGenerator(
    rotation_range=25,
    zoom_range=0.15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest")

valAug = ImageDataGenerator()

mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean

valAug.mean = mean

headmodel = ResNet50(weights="imagenet", include_top=False,
                  input_tensor=Input(shape=(224, 224, 3)))

model = headmodel.output
model = GlobalAveragePooling2D()(model)
model = Flatten(name="flatten")(model)
model = Dense(512, activation="relu")(model)
model = Dropout(0.4)(model)
model = Dense(len(lb.classes_), activation="softmax")(model)

moodel = Model(inputs=headmodel.input, outputs=model)

for layer in headmodel.layers:
    layer.trainable = False

# Phase 1: train top layers with frozen backbone
opt = Adam(learning_rate=1e-3)
moodel.compile(loss="categorical_crossentropy", optimizer=opt,
               metrics=["accuracy"])

checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', 
                            save_best_only=True, mode='max', verbose=1)
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, 
                          restore_best_weights=True, verbose=1)

batch_size = 32

H1 = moodel.fit(
    trainAug.flow(trainX, trainY, batch_size=batch_size),
    steps_per_epoch=max(1, len(trainX) // batch_size),
    validation_data=valAug.flow(testX, testY, batch_size=batch_size),
    validation_steps=max(1, len(testX) // batch_size),
    epochs=12,
    callbacks=[checkpoint, early_stop])

# Phase 2: fine-tune last blocks of ResNet50
for layer in headmodel.layers[-30:]:
    layer.trainable = True

opt_ft = Adam(learning_rate=1e-4)
moodel.compile(loss="categorical_crossentropy", optimizer=opt_ft,
               metrics=["accuracy"])

early_stop_ft = EarlyStopping(monitor='val_accuracy', patience=3,
                              restore_best_weights=True, verbose=1)

H2 = moodel.fit(
    trainAug.flow(trainX, trainY, batch_size=batch_size),
    steps_per_epoch=max(1, len(trainX) // batch_size),
    validation_data=valAug.flow(testX, testY, batch_size=batch_size),
    validation_steps=max(1, len(testX) // batch_size),
    epochs=8,
    callbacks=[checkpoint, early_stop_ft])

predictions = moodel.predict(testX, batch_size=batch_size)
y_true = testY.argmax(axis=1)
y_pred = predictions.argmax(axis=1)

report = classification_report(y_true, y_pred, target_names=lb.classes_, output_dict=True)
print(classification_report(y_true, y_pred, target_names=lb.classes_))

# Save detailed report
with open("classification_report.json", "w") as f:
    import json
    json.dump(report, f, indent=2)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
np.savetxt("confusion_matrix.csv", cm, fmt='%d', delimiter=',')


# save the model and labels to disk
moodel.save("model.h5")

with open("lb.pickle", "wb") as f:
    f.write(pickle.dumps(lb))
