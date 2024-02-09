from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
# from model_utils import process_video, crop_and_resize, load_model, predict, generate_heatmaps
import cv2
from mtcnn.mtcnn import MTCNN
import torch
from torchvision import models, datasets, transforms
from torchcam.methods import GradCAM
from torch.utils.data import DataLoader
import torch.nn as nn
from PIL import Image
from io import BytesIO
import base64
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights

app = Flask(__name__)

# Configurations
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['FRAME_FOLDER'] = 'frames'
app.config['CROPPED_FOLDER'] = 'cropped'
app.config['HEATMAP_FOLDER'] = 'static/heatmaps'  # Define the heatmap folder
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'mov', 'avi'}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Define the device

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        file = request.files.get('video')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)
            
            # Process video
            frame_output_dir = os.path.join(app.config['FRAME_FOLDER'], os.path.splitext(filename)[0])
            video_to_frames(video_path, frame_output_dir)
            
            # Crop and resize faces
            cropped_output_dir = os.path.join(app.config['CROPPED_FOLDER'], os.path.splitext(filename)[0])
            crop_and_resize(frame_output_dir, cropped_output_dir)
            
            # Load model and make predictions
            model = load_model('Resnet18_model.pth')
            test_loader = create_test_loader(cropped_output_dir)
            predictions = predict(model, test_loader)
            
            # Generate heatmaps
            heatmap_paths = generate_heatmaps(model, test_loader)
            
            return render_template('result.html', predictions=predictions, heatmap_paths=heatmap_paths)

    return render_template('index.html')

# Route to serve heatmap images
@app.route('/heatmap/<path:filename>')
def heatmap(filename):
    return send_from_directory(app.config['HEATMAP_FOLDER'], filename) 

# Helper functions for video processing
def video_to_frames(video_path, output_dir):
    # Open the video file
    subsample_rate = 1
    video_capture = cv2.VideoCapture(video_path)

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the video's frame rate
    frame_rate = int(video_capture.get(cv2.CAP_PROP_FPS))

    frame_count = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break  # Break the loop if no more frames are available
        
        if frame_count % (frame_rate // subsample_rate) == 0:
            output_frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(output_frame_path, frame)
        frame_count += 1

    video_capture.release()
    print(f"{frame_count // (frame_rate // subsample_rate)} frames extracted from {os.path.basename(video_path)}")


def crop_and_resize(input_dir, output_dir, size=(224, 224)):
    detector = MTCNN()

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        # os.makedirs(output_dir)
        os.makedirs(output_dir+"/all_images")

        
    for img_file in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_file)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        # Detect faces in the image
        faces = detector.detect_faces(np.array(img))
        if faces:
            x, y, width, height = faces[0]['box']
            face = img.crop((x, y, x + width, y + height))
            face = face.resize(size)
            face.save(os.path.join(output_dir,"all_images", img_file))
    pass

def create_test_loader(cropped_output_dir):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root=cropped_output_dir, transform=transform)
    return DataLoader(dataset, batch_size=32, shuffle=False)

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")

    # Load the saved model or state dict
    loaded = torch.load(model_path, map_location=torch.device('cpu'))

    if isinstance(loaded, dict):
        # If the loaded object is a state dictionary
        model = models.resnet18(weights=None)  # Initialize model without pre-trained weights
        model.load_state_dict(loaded)
    else:
        # If the loaded object is a complete model
        model = loaded

    model.eval()
    return model



def predict(model, test_loader):
    class_names = ['altered_frames_cropped', 'original_frames_cropped']

    from collections import Counter

    predictions = []
    with torch.no_grad():
        for data, _ in test_loader:
            # Move data to the same device as the model
            data = data.to(device)

            # Forward pass
            output = model(data)
            # print(output)
            # Convert output probabilities to class predictions (0 or 1)
            predicted_classes = (output > 0.0).int()
            predictions.extend(predicted_classes.cpu().numpy())

    # Flatten the list of predictions
    flat_predictions = [item for sublist in predictions for item in sublist]

    prediction_counts = Counter(flat_predictions)

    most_common_prediction = prediction_counts.most_common(1)[0][0]
    if most_common_prediction==1:
        prediction = 'Predicted Class: Original Image'
    else:
        prediction = 'Predicted Class: Altered Image'
    return prediction




import os
from torchvision import transforms
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from PIL import Image
from torchvision.transforms.functional import to_pil_image

def generate_heatmaps(model, test_loader, num_images=5):
    model.eval()  # Set the model to evaluation mode
    
    target_layer = 'layer4'  # Define the target layer for Grad-CAM
    
    # Ensure that gradients are enabled for the target layer
    for param in getattr(model, target_layer).parameters():
        param.requires_grad = True

    cam_extractor = GradCAM(model, target_layer)  # Create a Grad-CAM object
    
    # Define the image preprocessing
    preprocess = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    heatmap_paths = []  # Initialize a list to store paths to heatmap images
    collected_images = 0  # Counter for the number of images processed
    
    # Iterate over the dataset to process each image
    for images, _ in test_loader:
        for i in range(images.size(0)):
            if collected_images >= num_images:  # Stop after reaching the desired number of images
                break

            image = images[i]  # Get the i-th image in the batch
            input_tensor = preprocess(image).unsqueeze(0).to(device)  # Preprocess and add batch dimension

            # Generate CAM
            with torch.set_grad_enabled(True):
                outputs = model(input_tensor)
                activation_map = cam_extractor(outputs.squeeze().argmax().item(), outputs)
                if isinstance(activation_map, list):  # Check if the activation_map is a list
                    activation_map = activation_map[0]  # Get the first item if it's a list

            # Move the activation map to CPU and convert to numpy
            activation_map_np = activation_map.squeeze().cpu().numpy()

            # Convert the activation map to PIL image
            activation_map_pil = transforms.ToPILImage()(activation_map_np)

            # Resize the CAM and overlay it
            result = overlay_mask(transforms.ToPILImage()(image), activation_map_pil, alpha=0.5)
            # Save the heatmap overlay image
            heatmap_path = os.path.join(app.config['HEATMAP_FOLDER'], f'heatmap_{collected_images}.png')
            result.save(heatmap_path)  # Save the result to the specified path
            heatmap_paths.append(heatmap_path)  # Append the path to the list

            collected_images += 1  # Increment the image counter

        if collected_images >= num_images:  # Break the outer loop if we've reached the desired count
            break

    # Disable gradients for the target layer
    for param in getattr(model, target_layer).parameters():
        param.requires_grad = False

    return heatmap_paths





if __name__ == '__main__':
    app.run(debug=True)
