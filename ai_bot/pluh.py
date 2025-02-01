from tensorflow.python.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
def get_class(image, model, labels):
  np.set_printoptions(suppress=True)
  model = load_model(model, compile=False)
  class_names = open(labels, "r").readlines()

  data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

  image = image.convert("RGB")

  size = (224, 224)
  image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

  image_array = np.asarray(image)

  normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

  data[0] = normalized_image_array

  # Memprediksi model
  prediction = model.predict(data)
  index = np.argmax(prediction)
  class_name = class_names[index]
  confidence_score = prediction[0][index]

  return class_name[2:], confidence_score