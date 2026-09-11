"""
Arquitectura U-Net -- misma estructura que create_unet_model() del
notebook original: 5 bloques de encoder, 4 de decoder con skip
connections, kernels 5x5 en encoder y 3x3 en decoder.
"""
from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, Dropout, Input,
                                      MaxPooling2D, concatenate)
from tensorflow.keras.models import Model


def create_unet_model(image_size: int, num_classes: int, base_filters: int = 128) -> Model:
    f1 = base_filters
    f2 = base_filters // 2
    f3 = base_filters // 4
    f4 = base_filters // 8
    f5 = base_filters // 16

    # ===============
    # Entrada
    Image_input = Input(shape=(image_size, image_size, 1))

    # ===============
    # Codificador

    # conv1
    conv1 = Conv2D(f1, (5, 5), activation='relu', padding='same')(Image_input)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(f1, (5, 5), activation='relu', padding='same')(conv1)
    maxp1 = MaxPooling2D((2, 2))(conv1)

    # conv2
    conv2 = Conv2D(f2, (5, 5), activation='relu', padding='same')(maxp1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(f2, (5, 5), activation='relu', padding='same')(conv2)
    maxp2 = MaxPooling2D((2, 2))(conv2)

    # conv3
    conv3 = Conv2D(f3, (5, 5), activation='relu', padding='same')(maxp2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(f3, (5, 5), activation='relu', padding='same')(conv3)
    maxp3 = MaxPooling2D((2, 2))(conv3)

    # conv4
    conv4 = Conv2D(f4, (5, 5), activation='relu', padding='same')(maxp3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(f4, (5, 5), activation='relu', padding='same')(conv4)
    maxp4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # conv5
    conv5 = Conv2D(f5, (5, 5), activation='relu', padding='same')(maxp4)
    conv5 = Dropout(0.3)(conv5)
    conv5 = Conv2D(f5, (5, 5), activation='relu', padding='same')(conv5)

    # ===============
    # Decodificador

    # dec1
    dec1 = Conv2DTranspose(f4, (2, 2), strides=(2, 2), padding='same')(conv5)
    dec1 = concatenate([dec1, conv4])
    dec1 = Conv2D(f4, (3, 3), activation='relu', padding='same')(dec1)
    dec1 = Dropout(0.2)(dec1)
    dec1 = Conv2D(f4, (3, 3), activation='relu', padding='same')(dec1)

    # dec2
    dec2 = Conv2DTranspose(f3, (2, 2), strides=(2, 2), padding='same')(dec1)
    dec2 = concatenate([dec2, conv3])
    dec2 = Conv2D(f3, (3, 3), activation='relu', padding='same')(dec2)
    dec2 = Dropout(0.2)(dec2)
    dec2 = Conv2D(f3, (3, 3), activation='relu', padding='same')(dec2)

    # dec3
    dec3 = Conv2DTranspose(f2, (2, 2), strides=(2, 2), padding='same')(dec2)
    dec3 = concatenate([dec3, conv2])
    dec3 = Conv2D(f2, (3, 3), activation='relu', padding='same')(dec3)
    dec3 = Dropout(0.2)(dec3)
    dec3 = Conv2D(f2, (3, 3), activation='relu', padding='same')(dec3)

    # dec4
    dec4 = Conv2DTranspose(f1, (2, 2), strides=(2, 2), padding='same')(dec3)
    dec4 = concatenate([dec4, conv1], axis=3)
    dec4 = Conv2D(f1, (3, 3), activation='relu', padding='same')(dec4)
    dec4 = Dropout(0.2)(dec4)
    dec4 = Conv2D(f1, (3, 3), activation='relu', padding='same')(dec4)

    # ===============
    # Salida

    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(dec4)

    # ===============
    # Interconectar todo en un modelo

    unet = Model(inputs=Image_input, outputs=outputs)
    unet.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return unet