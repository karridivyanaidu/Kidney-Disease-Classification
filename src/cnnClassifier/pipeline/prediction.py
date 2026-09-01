import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


class PredictionPipeline:

    def __init__(self, filename):
        self.filename = filename

        self.model = load_model(
            os.path.join("model", "model.h5"),
            compile=False
        )

    def predict(self):

        test_image = image.load_img(
            self.filename,
            target_size=(224, 224)
        )

        test_image = image.img_to_array(test_image)
        test_image = test_image / 255.0
        test_image = np.expand_dims(test_image, axis=0)

        result = np.argmax(
            self.model.predict(
                test_image,
                verbose=0
            ),
            axis=1
        )

        if result[0] == 1:
            prediction = "Tumor"
        else:
            prediction = "Normal"

        return [{"image": prediction}]