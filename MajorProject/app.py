import numpy as np
import torch
from flask import Flask, request, jsonify, render_template, Response
from utilities import transform, filter_bboxes_from_outputs, get_image_region, plot_results, regions
from transformers import (
    DetrForObjectDetection,
    DPTForDepthEstimation,
    DPTFeatureExtractor,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
)
from PIL import Image

# Initializing a vision encoder-decoder model for image captioning
ve_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Initializing a feature extractor for the Vision Transformer (ViT)
# This will be used to process images and extract relevant features
feature_extractor_vit = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Initializing a tokenizer for text input
# This will convert raw text into tokenized input that the model can understand
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

app = Flask(__name__)
detrmodel = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
dptmodel = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
detrmodel.eval()
dptmodel.eval()


# Home route
@app.route("/")
def home():
    return render_template("index2.html")


@app.route("/upload")
def photo():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    priority_list = []
    file = request.files["data"]
    file.save("static/uploaded_image.jpg")
    # Read the image via file.stream
    im = Image.open(file.stream)
    img = transform(im).unsqueeze(0)

    outputs = detrmodel(img)
    probas_to_keep, bboxes_scaled, final_outputs = filter_bboxes_from_outputs(im, outputs, threshold=0.95)
    plot_results(im, probas_to_keep, bboxes_scaled)

    object_list = [[int(item) if isinstance(item, float) else item for item in sublist] for sublist in final_outputs]

    pixel_values = feature_extractor(im, return_tensors="pt").pixel_values
    with torch.no_grad():
        outputs = dptmodel(pixel_values)
        predicted_depth = outputs.predicted_depth
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=im.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    output = prediction.cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    # cv2.imread('/content/Depth Image', cv2.IMREAD_GRAYSCALE)
    depth_image = formatted
    # Perform object detection or segmentation on the normal image to obtain object bounding boxes and names
    # and store them in a list called 'objects' with elements of the form: (object_name, x, y, w, h)
    objects = object_list
    # Calculate average depth value for each object
    objects_with_depth = []
    for idx, obj in enumerate(objects):
        (
            object_name,
            x,
            y,
            w,
            h,
            origin,
        ) = obj  # Object bounding box coordinates and name
        depth_values = depth_image[y : y + h, x : x + w]
        average_depth = np.mean(depth_values)
        objects_with_depth.append(([object_name, idx, origin], average_depth))

    # Sort objects based on depth (lower depth values have higher priority)
    objects_with_depth.sort(key=lambda x: x[1])

    # Create a priority list of objects with object names and initial indices
    priority_list = [obj[0] for obj in objects_with_depth]
    priority_list = sorted(priority_list, key=lambda x: x[1])

    print("-" * 20)
    print(f"The priority list is: \n{priority_list}")
    print("-" * 20)

    for i in priority_list:
        center = i[2]
        i.insert(3, get_image_region(center))

    grp = {}
    for k in regions.keys():
        grp[k] = []
    l = len(priority_list)
    delim = l if l < 5 else 5
    for i in priority_list[:delim]:
        grp[i[3]].append(i[0])

    answer = "The most Priority objects are :"
    for k, v in grp.items():
        text = ""
        if len(v):
            for i in v:
                text += i
                text += " "
            text += " at " + k
            answer += text + ", "
    print(answer)

    return Response(answer)


app.run(debug=True)
