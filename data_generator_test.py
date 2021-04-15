import numpy as np
import tensorflow.keras
from PIL import Image
from TimeDistributedImageDataGenerator import TimeDistributedImageDataGenerator
from tqdm import tqdm


def generate_generator_multiple(generator, dir1, dir2, batch_size, img_height, img_width):
    genX1 = generator.flow_from_directory(dir1,
                                          target_size=(img_height, img_width),
                                          class_mode='categorical',
                                          batch_size=batch_size,
                                          shuffle=False,
                                          seed=7)

    genX2 = generator.flow_from_directory(dir2,
                                          target_size=(img_height, img_width),
                                          class_mode='categorical',
                                          batch_size=batch_size,
                                          shuffle=False,
                                          seed=7)
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        yield [X1i[0], X2i[0]], X2i[1]  # Yield both images and their mutual label


if __name__ == "__main__":
    datagen = TimeDistributedImageDataGenerator.TimeDistributedImageDataGenerator(time_steps=5)



