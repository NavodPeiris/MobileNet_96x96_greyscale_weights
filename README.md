#### This repo contain weights of MobileNetV1, MobileNetV2 models trained on 96x96 greyscale images of ImageNet dataset. suitable for a transfer learning model for image classification, object detection tasks. 

#### These are Lightweight, energy efficient and memory efficient models that can be deployed on Edge devices such as Microcontrollers.

#### How to use with MobileNetV1:
```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Input
)

from tensorflow.keras.models import Model
from keras.applications.mobilenet import MobileNet
from tensorflow.keras.optimizers import Adam

epochs = 50
optimizer = Adam(learning_rate=0.0005)   # use any learning rate

input_tensor = Input(shape=(96, 96, 1))
mobilenet_model = MobileNet(
    input_shape=(96, 96, 1),
    input_tensor=input_tensor, 
    pooling="avg", 
    alpha=0.25,   # 0.25, 0.2, 0.1 
    weights="mobilenetV1_0.25_96x96_greyscale_weights.h5", # 0.25, 0.2, 0.1
    include_top=False
    )

mobilenet_model.trainable = False

mobilenet_output = mobilenet_model.output

# Dense layer
dense_layer = Dense(256, activation="relu")(mobilenet_output)

# Dropout layer
dropout_layer = Dropout(0.1)(dense_layer)

# classification layer
classification_layer = Dense(num_classes, activation='softmax')(dropout_layer)

model = Model(inputs=mobilenet_model.input, outputs=classification_layer)

print("Compiling model...")
model.compile(loss="categorical_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"])

model.summary()
```

#### How to use with MobileNetV2:
```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Input
)

from tensorflow.keras.models import Model
from keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.optimizers import Adam

epochs = 50
optimizer = Adam(learning_rate=0.0005)   # use any learning rate

input_tensor = Input(shape=(96, 96, 1))
mobilenet_model = MobileNetV2(
    input_shape=(96, 96, 1),
    input_tensor=input_tensor, 
    pooling="avg", 
    alpha=0.35,   # 0.35, 0.1, 0.05  
    weights="mobilenetV2_0.35_96x96_greyscale_weights.h5", # 0.35, 0.1, 0.05
    include_top=False
    )

mobilenet_model.trainable = False

mobilenet_output = mobilenet_model.output

# Dense layer
dense_layer = Dense(256, activation="relu")(mobilenet_output)

# Dropout layer
dropout_layer = Dropout(0.1)(dense_layer)

# classification layer
classification_layer = Dense(num_classes, activation='softmax')(dropout_layer)

model = Model(inputs=mobilenet_model.input, outputs=classification_layer)

print("Compiling model...")
model.compile(loss="categorical_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"])

model.summary()
```