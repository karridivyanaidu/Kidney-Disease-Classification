import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


class PredictionPipeline:

    def __init__(self, filename):
        self.filename = filename

        # Load deployment model only once
        self.model = load_model(
            os.path.join("model", "model.h5")
        )

    def predict(self):

        imagename = self.filename

        # Load image
        test_image = image.load_img(
            imagename,
            target_size=(224, 224)
        )

        # Convert image to array
        test_image = image.img_to_array(test_image)

        # Same preprocessing used during training
        test_image = test_image / 255.0

        # Add batch dimension
        test_image = np.expand_dims(test_image, axis=0)

        # Make prediction
        result = np.argmax(
            self.model.predict(test_image),
            axis=1
        )

        print(result)

        # Convert result to class
        if result[0] == 1:
            prediction = "Tumor"
        else:
            prediction = "Normal"

        return [{"image": prediction}]