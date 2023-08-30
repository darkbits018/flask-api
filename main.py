from flask import Flask, request, jsonify
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from PIL import Image
import numpy as np
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel
import torch

app = Flask(__name__)

# Load MobileNetV2 model
object_recognition_model = MobileNetV2(weights='imagenet')

# Load VisionEncoderDecoderModel
image_captioning_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Variable to store the response
stored_response = None

@app.route("/")
def home_view():
        return "<h1>Hello World!</h1>"


@app.route('/recognize', methods=['POST'])
def process_image():
    global stored_response

    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files['image']
    pil_image = Image.open(image.stream)

    # Object Recognition
    object_name, confidence = recognize_image(pil_image)
    description = generate_description(object_name)

    # Image Captioning
    image_caption = generate_image_caption(pil_image)

    stored_response = {
        "prediction": object_name,
        "confidence": float(confidence),
        "object_description": description,
        "image_caption": image_caption
    }

    return jsonify(stored_response), 200


@app.route('/get_response', methods=['GET'])
def get_response():
    global stored_response

    if stored_response:
        return jsonify(stored_response), 200
    else:
        return 'No response stored', 404


def recognize_image(image):
    image = image.resize((224, 224))  # Resize image to match MobileNetV2 input size
    image = preprocess_input(np.array(image)[np.newaxis, ...])

    predictions = object_recognition_model.predict(image)
    decoded_predictions = decode_predictions(predictions)[0]

    top_prediction = decoded_predictions[0][1]
    confidence = decoded_predictions[0][2]

    return top_prediction, confidence


def generate_description(object_name):
    return f"This is an image of a {object_name}."


def generate_image_caption(image):
    pixel_values = image_processor(image, return_tensors="pt").pixel_values
    generated_ids = image_captioning_model.generate(pixel_values, max_new_tokens=30)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

if __name__ == "__main__":
        app.run()

