{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "4b2197e9-0ae8-48ca-8c86-5e7882f95969",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import tensorflow as tf\n",
    "from tensorflow import keras\n",
    "from tensorflow.keras import layers\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.metrics import classification_report, confusion_matrix\n",
    "import numpy as np\n",
    "import itertools"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "7c19e97f-cfc1-41e7-863f-8d825e26c8c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "DATA_DIR = r\"C:\\Python\\Dataset\\testing dataset\"   \n",
    "TRAIN_SUBDIR = \"train\"\n",
    "VAL_SUBDIR = \"val\"\n",
    "MODEL_DIR = r\"C:\\Python\\model\"\n",
    "TFLITE_PATH = os.path.join(MODEL_DIR, \"cattle_breed.tflite\")\n",
    "\n",
    "IMG_SIZE = (224, 224)\n",
    "BATCH_SIZE = 16\n",
    "EPOCHS = 15\n",
    "NUM_CLASSES = 6  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "0d46aec6-21f7-417c-9042-f6acddaedaf0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 2683 files belonging to 6 classes.\n",
      "Found 299 files belonging to 6 classes.\n"
     ]
    }
   ],
   "source": [
    "# === Load dataset ===\n",
    "train_dir = os.path.join(DATA_DIR, TRAIN_SUBDIR)\n",
    "val_dir = os.path.join(DATA_DIR, VAL_SUBDIR)\n",
    "\n",
    "train_ds = tf.keras.utils.image_dataset_from_directory(\n",
    "    train_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE\n",
    ")\n",
    "val_ds = tf.keras.utils.image_dataset_from_directory(\n",
    "    val_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "66c83e81-4848-4269-850f-c50d8a544a13",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "AUTOTUNE = tf.data.AUTOTUNE\n",
    "train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)\n",
    "val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "12e028ba-c76e-4888-aed6-02af18211ece",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "data_augmentation = keras.Sequential([\n",
    "    layers.RandomFlip(\"horizontal\"),\n",
    "    layers.RandomRotation(0.1),\n",
    "    layers.RandomZoom(0.1),\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "ce19a282-bf22-4dc5-8cfb-1121ed11e64a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# === MobileNetV3 backbone ===\n",
    "base_model = tf.keras.applications.MobileNetV3Large(\n",
    "    input_shape=IMG_SIZE + (3,),\n",
    "    include_top=False,\n",
    "    weights=\"imagenet\"\n",
    ")\n",
    "\n",
    "base_model.trainable = True\n",
    "for layer in base_model.layers[:-30]:  # freeze earlier layers\n",
    "    layer.trainable = False"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "ed9094e0-6b1d-49f5-9c51-1fdada9d9741",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "inputs = keras.Input(shape=IMG_SIZE + (3,))\n",
    "x = data_augmentation(inputs)\n",
    "x = tf.keras.applications.mobilenet_v3.preprocess_input(x)\n",
    "x = base_model(x, training=False)\n",
    "x = layers.GlobalAveragePooling2D()(x)\n",
    "x = layers.Dropout(0.3)(x)\n",
    "outputs = layers.Dense(NUM_CLASSES, activation=\"softmax\")(x)\n",
    "\n",
    "model = keras.Model(inputs, outputs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "eceb2322-bf69-48d3-8f1e-e7b1132dec76",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\">Model: \"functional_3\"</span>\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1mModel: \"functional_3\"\u001b[0m\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓\n",
       "┃<span style=\"font-weight: bold\"> Layer (type)                         </span>┃<span style=\"font-weight: bold\"> Output Shape                </span>┃<span style=\"font-weight: bold\">         Param # </span>┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩\n",
       "│ input_layer_4 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">InputLayer</span>)           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">224</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">224</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">3</span>)         │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ sequential_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Sequential</span>)            │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">224</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">224</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">3</span>)         │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ MobileNetV3Large (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Functional</span>)        │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">7</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">7</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">960</span>)           │       <span style=\"color: #00af00; text-decoration-color: #00af00\">2,996,352</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ global_average_pooling2d_1           │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">960</span>)                 │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "│ (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">GlobalAveragePooling2D</span>)             │                             │                 │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dropout_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dropout</span>)                  │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">960</span>)                 │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">6</span>)                   │           <span style=\"color: #00af00; text-decoration-color: #00af00\">5,766</span> │\n",
       "└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘\n",
       "</pre>\n"
      ],
      "text/plain": [
       "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓\n",
       "┃\u001b[1m \u001b[0m\u001b[1mLayer (type)                        \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1mOutput Shape               \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1m        Param #\u001b[0m\u001b[1m \u001b[0m┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩\n",
       "│ input_layer_4 (\u001b[38;5;33mInputLayer\u001b[0m)           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m224\u001b[0m, \u001b[38;5;34m224\u001b[0m, \u001b[38;5;34m3\u001b[0m)         │               \u001b[38;5;34m0\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ sequential_1 (\u001b[38;5;33mSequential\u001b[0m)            │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m224\u001b[0m, \u001b[38;5;34m224\u001b[0m, \u001b[38;5;34m3\u001b[0m)         │               \u001b[38;5;34m0\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ MobileNetV3Large (\u001b[38;5;33mFunctional\u001b[0m)        │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m7\u001b[0m, \u001b[38;5;34m7\u001b[0m, \u001b[38;5;34m960\u001b[0m)           │       \u001b[38;5;34m2,996,352\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ global_average_pooling2d_1           │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m960\u001b[0m)                 │               \u001b[38;5;34m0\u001b[0m │\n",
       "│ (\u001b[38;5;33mGlobalAveragePooling2D\u001b[0m)             │                             │                 │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dropout_1 (\u001b[38;5;33mDropout\u001b[0m)                  │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m960\u001b[0m)                 │               \u001b[38;5;34m0\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense_1 (\u001b[38;5;33mDense\u001b[0m)                      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m6\u001b[0m)                   │           \u001b[38;5;34m5,766\u001b[0m │\n",
       "└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Total params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">3,002,118</span> (11.45 MB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Total params: \u001b[0m\u001b[38;5;34m3,002,118\u001b[0m (11.45 MB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">1,600,486</span> (6.11 MB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Trainable params: \u001b[0m\u001b[38;5;34m1,600,486\u001b[0m (6.11 MB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Non-trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">1,401,632</span> (5.35 MB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Non-trainable params: \u001b[0m\u001b[38;5;34m1,401,632\u001b[0m (5.35 MB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "\n",
    "model.compile(\n",
    "    optimizer=keras.optimizers.Adam(1e-4),\n",
    "    loss=\"sparse_categorical_crossentropy\",\n",
    "    metrics=[\"accuracy\"]\n",
    ")\n",
    "\n",
    "model.summary()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "dfb03065-c7ec-45d0-9d36-dac6d1945309",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/15\n",
      "\u001b[1m168/168\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m73s\u001b[0m 336ms/step - accuracy: 0.4558 - loss: 1.4469 - val_accuracy: 0.7124 - val_loss: 0.7940\n",
      "Epoch 2/15\n",
      "\u001b[1m168/168\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m52s\u001b[0m 309ms/step - accuracy: 0.7518 - loss: 0.7392 - val_accuracy: 0.8127 - val_loss: 0.5561\n",
      "Epoch 3/15\n",
      "\u001b[1m168/168\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m43s\u001b[0m 253ms/step - accuracy: 0.8207 - loss: 0.5462 - val_accuracy: 0.8495 - val_loss: 0.4570\n",
      "Epoch 4/15\n",
      "\u001b[1m168/168\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m39s\u001b[0m 234ms/step - accuracy: 0.8528 - loss: 0.4343 - val_accuracy: 0.8829 - val_loss: 0.3778\n",
      "Epoch 5/15\n",
      "\u001b[1m168/168\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m39s\u001b[0m 229ms/step - accuracy: 0.8871 - loss: 0.3204 - val_accuracy: 0.8997 - val_loss: 0.3412\n",
      "Epoch 6/15\n",
      "\u001b[1m168/168\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m39s\u001b[0m 231ms/step - accuracy: 0.9161 - loss: 0.2461 - val_accuracy: 0.8829 - val_loss: 0.3562\n",
      "Epoch 7/15\n",
      "\u001b[1m168/168\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m42s\u001b[0m 247ms/step - accuracy: 0.9340 - loss: 0.1981 - val_accuracy: 0.8963 - val_loss: 0.3306\n",
      "Epoch 8/15\n",
      "\u001b[1m168/168\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m44s\u001b[0m 264ms/step - accuracy: 0.9478 - loss: 0.1593 - val_accuracy: 0.9130 - val_loss: 0.2824\n",
      "Epoch 9/15\n",
      "\u001b[1m168/168\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m55s\u001b[0m 328ms/step - accuracy: 0.9605 - loss: 0.1296 - val_accuracy: 0.9130 - val_loss: 0.2744\n",
      "Epoch 10/15\n",
      "\u001b[1m168/168\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m57s\u001b[0m 339ms/step - accuracy: 0.9724 - loss: 0.0944 - val_accuracy: 0.8796 - val_loss: 0.3638\n",
      "Epoch 11/15\n",
      "\u001b[1m168/168\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m51s\u001b[0m 305ms/step - accuracy: 0.9810 - loss: 0.0720 - val_accuracy: 0.9164 - val_loss: 0.2698\n",
      "Epoch 12/15\n",
      "\u001b[1m168/168\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m41s\u001b[0m 243ms/step - accuracy: 0.9892 - loss: 0.0551 - val_accuracy: 0.8896 - val_loss: 0.3799\n",
      "Epoch 13/15\n",
      "\u001b[1m168/168\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m40s\u001b[0m 237ms/step - accuracy: 0.9884 - loss: 0.0444 - val_accuracy: 0.9064 - val_loss: 0.3310\n",
      "Epoch 14/15\n",
      "\u001b[1m168/168\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m42s\u001b[0m 249ms/step - accuracy: 0.9911 - loss: 0.0369 - val_accuracy: 0.9064 - val_loss: 0.3208\n",
      "Epoch 15/15\n",
      "\u001b[1m168/168\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m46s\u001b[0m 275ms/step - accuracy: 0.9929 - loss: 0.0331 - val_accuracy: 0.9064 - val_loss: 0.3099\n"
     ]
    }
   ],
   "source": [
    "\n",
    "history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "ee1abf2b-4b59-42d5-8218-540f066082bc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Saved model to C:\\Python\\model\\cattle_breed.keras\n"
     ]
    }
   ],
   "source": [
    "\n",
    "keras_path = os.path.join(MODEL_DIR, \"cattle_breed.keras\")\n",
    "model.save(keras_path, include_optimizer=False)\n",
    "print(f\"Saved model to {keras_path}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "bdca7648-aabf-40da-b650-f477d7ee467a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: C:\\Users\\kingr\\AppData\\Local\\Temp\\tmp3y_ke49w\\assets\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Assets written to: C:\\Users\\kingr\\AppData\\Local\\Temp\\tmp3y_ke49w\\assets\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Saved artifact at 'C:\\Users\\kingr\\AppData\\Local\\Temp\\tmp3y_ke49w'. The following endpoints are available:\n",
      "\n",
      "* Endpoint 'serve'\n",
      "  args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name='input_layer_4')\n",
      "Output Type:\n",
      "  TensorSpec(shape=(None, 6), dtype=tf.float32, name=None)\n",
      "Captures:\n",
      "  1894597714416: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597713536: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597716880: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597714592: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597715296: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597721280: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597805504: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597808672: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597718816: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597804624: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597809024: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597810432: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597814304: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597809552: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597812016: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597818000: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597819232: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597816416: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597820112: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597819936: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597599376: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597596912: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597601488: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597598496: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597599904: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597604656: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597605008: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597604304: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597606944: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597986960: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597989072: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597988016: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597990832: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597989952: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597992064: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597995936: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597997520: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597999984: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597997872: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597996640: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894597999280: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598838576: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598841920: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598838928: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598840864: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598847552: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598845616: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598849136: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598848784: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598847904: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598848256: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598969648: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598975632: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598970528: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598973520: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598982320: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598977744: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598983200: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598973696: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598980032: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894599054208: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894599058784: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894599055792: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894599057200: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894599062832: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894599060368: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894599064944: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894599061952: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894599063360: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894599065120: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598152208: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598155024: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598151856: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598153440: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598161712: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598159776: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598163648: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598154848: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598161008: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598218624: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598223200: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598220208: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598221616: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598228832: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598227776: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598225664: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598229184: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598228128: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598282752: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598284864: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598287680: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598284512: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598286096: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598294368: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598292608: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598287504: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598382112: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598386512: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598384048: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598388624: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598385632: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598387040: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598392320: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598392672: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598477952: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598393552: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598477600: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598485520: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598483056: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598487632: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598484640: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598486048: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598491152: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598558816: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598560400: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598557760: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598559344: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598564976: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598562512: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598567088: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598564096: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598565504: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598572544: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598571488: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598569376: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598572896: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598573424: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598662224: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598659760: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598664336: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598661344: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598662752: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598669968: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598667504: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598672080: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598669088: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598670496: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598669792: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594344416: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594349344: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594344064: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594347760: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594354800: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594352336: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594356912: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594353920: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594355328: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594354624: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594212816: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594217392: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594214400: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594215808: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594222848: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594220384: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594224960: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594221968: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594223376: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594222672: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594325744: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594330848: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594329440: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594328560: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594335424: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594334368: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594338592: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594336304: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594337008: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594335248: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594423872: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594428448: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594424752: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594426864: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594435312: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594433376: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594437248: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594437952: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594438128: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594492224: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594496800: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594493808: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594495216: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594500848: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594498384: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594502960: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594499968: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594501376: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594500672: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594604448: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594609024: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594605328: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594607440: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594615888: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594613952: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594617824: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594618528: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594618704: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594689184: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594693760: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594690768: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594692176: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594699392: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594696928: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594701504: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594698512: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594699920: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594699216: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594771808: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594776384: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594773392: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594774800: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594783952: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594781488: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594776208: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594866592: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594875392: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594872928: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594877504: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594874512: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594875920: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594882256: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594880496: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594878384: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594881904: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594880848: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594954848: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594952384: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594956960: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594953968: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594955376: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594961888: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594962592: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896594956784: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896595047168: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896595055968: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896595053504: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896595058080: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896595055088: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896595056496: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896595061776: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896595129440: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896595131024: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896595128384: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896595129968: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896595137008: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896595134544: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896595139120: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896595136128: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896595137536: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896595144048: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896595136832: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896595138944: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896595243776: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896595254512: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896595252048: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896595256624: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896595253632: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896595255040: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896595254336: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896595358288: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896595362864: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896595359872: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1896595361280: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598769328: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "  1894598768272: TensorSpec(shape=(), dtype=tf.resource, name=None)\n",
      "TFLite model saved to C:\\Python\\model\\cattle_breed.tflite\n"
     ]
    }
   ],
   "source": [
    "\n",
    "def convert_to_tflite(keras_model_path, tflite_output_path):\n",
    "    model = keras.models.load_model(keras_model_path)\n",
    "    converter = tf.lite.TFLiteConverter.from_keras_model(model)\n",
    "    converter.optimizations = [tf.lite.Optimize.DEFAULT]\n",
    "    tflite_model = converter.convert()\n",
    "    with open(tflite_output_path, \"wb\") as f:\n",
    "        f.write(tflite_model)\n",
    "    print(f\"TFLite model saved to {tflite_output_path}\")\n",
    "\n",
    "convert_to_tflite(keras_path, TFLITE_PATH)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "2cbc4e84-ce3e-4a75-ae9e-f9321ebd11cb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAzoAAAHDCAYAAADss29MAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjYsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvq6yFwwAAAAlwSFlzAAAPYQAAD2EBqD+naQAAk6NJREFUeJzt3Qd4FNXXBvA3vRdCGqQQeu9NukhTEQVBqoAoWLFhgw/FLlbUvyIogoJIEQREQRARVKT33ktCICQhvbf9nnMnGxJIIGWTbe/veYYt2Z1cwpKZM/fcc2x0Op0OREREREREFsTW2AMgIiIiIiIyNAY6RERERERkcRjoEBERERGRxWGgQ0REREREFoeBDhERERERWRwGOkREREREZHEY6BARERERkcVhoENERERERBaHgQ4REREREVkcBjpERERERGRxGOiQRfvqq69gY2ODjh07GnsoREREBvP999+r49vu3buNPRQik8VAhyzajz/+iLCwMOzcuROnT5829nCIiIiIqIow0CGLde7cOWzduhUzZsyAn5+fCnpMUWpqqrGHQERERGRxGOiQxZLAplq1aujfvz+GDBlSbKCTkJCA559/Xs36ODk5ITg4GGPGjEFsbGzBazIyMvDGG2+gQYMGcHZ2Ro0aNXD//ffjzJkz6uubN29W6QNyW9j58+fV85JeoPfQQw/B3d1dvffuu++Gh4cHRo0apb7277//4oEHHkBoaKgaS0hIiBpbenr6DeM+fvw4hg4dqgI4FxcXNGzYEFOnTlVf27Rpk/q+K1euvOF9ixYtUl/btm1bhX62RERk+vbt24e77roLnp6e6tjTq1cvbN++vchrsrOz8eabb6J+/frqGFe9enV07doVGzZsKHhNVFQUxo0bp46RcnyS4+B9992njnNEpsze2AMgqiwS2EhA4ujoiBEjRmDWrFnYtWsX2rdvr76ekpKCbt264dixY3j44YfRpk0bFeCsXr0aFy9ehK+vL3Jzc3HPPfdg48aNGD58OJ599lkkJyerA8Dhw4dRt27dMo8rJycH/fr1UweSjz/+GK6urur5ZcuWIS0tDU888YQ60Ei63RdffKHGIl/TO3jwoBq3g4MDHn30URWkSeD066+/4t1338Xtt9+ugiT5+w8aNOiGn4mMuVOnThX++RIRkek6cuSIOlZIkPPyyy+rY8bXX3+tjhF/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",
      "text/plain": [
       "<Figure size 1000x500 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "\n",
    "acc = history.history[\"accuracy\"]\n",
    "val_acc = history.history[\"val_accuracy\"]\n",
    "loss = history.history[\"loss\"]\n",
    "val_loss = history.history[\"val_loss\"]\n",
    "\n",
    "plt.figure(figsize=(10, 5))\n",
    "plt.subplot(1, 2, 1)\n",
    "plt.plot(acc, label=\"Train Acc\")\n",
    "plt.plot(val_acc, label=\"Val Acc\")\n",
    "plt.legend()\n",
    "plt.title(\"Accuracy\")\n",
    "\n",
    "plt.subplot(1, 2, 2)\n",
    "plt.plot(loss, label=\"Train Loss\")\n",
    "plt.plot(val_loss, label=\"Val Loss\")\n",
    "plt.legend()\n",
    "plt.title(\"Loss\")\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "5c869adf-cfcc-4880-a0d0-b38981e32b4a",
   "metadata": {},
   "outputs": [
    {
     "ename": "PermissionError",
     "evalue": "[Errno 13] Permission denied: 'C:\\\\Python\\\\Dataset'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mPermissionError\u001b[0m                           Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[22], line 6\u001b[0m\n\u001b[0;32m      3\u001b[0m class_names \u001b[38;5;241m=\u001b[39m \u001b[38;5;28msorted\u001b[39m(os\u001b[38;5;241m.\u001b[39mlistdir(train_dir))\n\u001b[0;32m      5\u001b[0m img_path \u001b[38;5;241m=\u001b[39m \u001b[38;5;124mr\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mC:\u001b[39m\u001b[38;5;124m\\\u001b[39m\u001b[38;5;124mPython\u001b[39m\u001b[38;5;124m\\\u001b[39m\u001b[38;5;124mDataset\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m----> 6\u001b[0m img \u001b[38;5;241m=\u001b[39m \u001b[43mtf\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mkeras\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mutils\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mload_img\u001b[49m\u001b[43m(\u001b[49m\u001b[43mimg_path\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mtarget_size\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mIMG_SIZE\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m      7\u001b[0m img_array \u001b[38;5;241m=\u001b[39m tf\u001b[38;5;241m.\u001b[39mkeras\u001b[38;5;241m.\u001b[39mutils\u001b[38;5;241m.\u001b[39mimg_to_array(img)\n\u001b[0;32m      8\u001b[0m img_array \u001b[38;5;241m=\u001b[39m np\u001b[38;5;241m.\u001b[39mexpand_dims(img_array, axis \u001b[38;5;241m=\u001b[39m \u001b[38;5;241m0\u001b[39m)\n",
      "File \u001b[1;32mc:\\python\\lib\\site-packages\\keras\\src\\utils\\image_utils.py:235\u001b[0m, in \u001b[0;36mload_img\u001b[1;34m(path, color_mode, target_size, interpolation, keep_aspect_ratio)\u001b[0m\n\u001b[0;32m    233\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28misinstance\u001b[39m(path, pathlib\u001b[38;5;241m.\u001b[39mPath):\n\u001b[0;32m    234\u001b[0m         path \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mstr\u001b[39m(path\u001b[38;5;241m.\u001b[39mresolve())\n\u001b[1;32m--> 235\u001b[0m     \u001b[38;5;28;01mwith\u001b[39;00m \u001b[38;5;28;43mopen\u001b[39;49m\u001b[43m(\u001b[49m\u001b[43mpath\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43mrb\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m)\u001b[49m \u001b[38;5;28;01mas\u001b[39;00m f:\n\u001b[0;32m    236\u001b[0m         img \u001b[38;5;241m=\u001b[39m pil_image\u001b[38;5;241m.\u001b[39mopen(io\u001b[38;5;241m.\u001b[39mBytesIO(f\u001b[38;5;241m.\u001b[39mread()))\n\u001b[0;32m    237\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n",
      "\u001b[1;31mPermissionError\u001b[0m: [Errno 13] Permission denied: 'C:\\\\Python\\\\Dataset'"
     ]
    }
   ],
   "source": [
    "train_dir =r\"C:\\Python\\Dataset\\testing dataset\\train\"\n",
    "\n",
    "class_names = sorted(os.listdir(train_dir))\n",
    "\n",
    "img_path = r\"C:\\Python\\Dataset\"\n",
    "img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)\n",
    "img_array = tf.keras.utils.img_to_array(img)\n",
    "img_array = np.expand_dims(img_array, axis = 0)\n",
    "img_array = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)\n",
    "\n",
    "pred = model.predict(img_array)[0]\n",
    "\n",
    "top3_idx = np.argsort(pred)[-3:][::-1]\n",
    "\n",
    "print(\" top 3 predictions:\")\n",
    "for i in top3_idx:\n",
    "    print(f\"{class_names[i]}: {pred[i]*100:.2f}%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "15b7ea93-933c-4a7b-8158-978f7659a5dc",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b715071b-01f4-4db2-87a8-6dc943b24ffa",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0a75ac22-d527-47e8-a0d8-4ba8f0ffd826",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
