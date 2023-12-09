from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('models/icon_classification_model_2.h5')

test_image_path = 'testing/save.jpg'
img = Image.open(test_image_path)
img = img.resize((50, 50))
img = img.convert('L')
img_array = np.array(img)
img_array = img_array.astype('float32') / 255.0

img_array = img_array.reshape(1, 50, 50, 1)

predictions = model.predict(img_array)

top3_indices = np.argsort(predictions[0])[-3:][::-1]

label_dict = np.load("data/label_dict.npy", allow_pickle=True).item()

top3_labels = [list(label_dict.keys())[i] for i in top3_indices]

top3_probabilities = predictions[0][top3_indices]

for label, prob in zip(top3_labels, top3_probabilities):
    print(f"Label: {label}, Probability: {prob:.4f}")