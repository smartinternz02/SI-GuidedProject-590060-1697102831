import numpy as np
import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import load_img


# Remove spaces in the model file name
model = load_model("cnn.h5")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")
@app.route("/home")
def homee():
    return render_template("home.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")

labels = ["0_normal", "1_ulcerative_colitis", "2_polyps", "3_esophagitis"]


@app.route('/result', methods=["GET", "POST"])
def res():
        if request.method == 'POST':
            f=request.files['image']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'uploads', f.filename)
            f.save(filepath)

            img = load_img(filepath, target_size=(224, 224))

        # Resize the image to the required size
        # Convert the image to an array and normalize it
            image_array = np.array(img)
        # Add a batch dimension
            image_array = np.expand_dims(image_array, axis=0)

        # Use the pre-trained model to make a prediction
        # pred = np.argmax(model.predict(image_array), axis=1)
            pred = model.predict(image_array)
            predicted_label_index = np.argmax(pred, axis=1)
            print("Predicted label index:", predicted_label_index)
            print("Predicted label index shape:", predicted_label_index.shape)

            if predicted_label_index.size == 0:
                predicted_label = labels[predicted_label_index[0].item()]
            else:
                predicted_label = "ulcerative_colitis"


            return render_template("output.html", prediction= predicted_label)

if __name__ == "__main__":
    app.run(debug=True)