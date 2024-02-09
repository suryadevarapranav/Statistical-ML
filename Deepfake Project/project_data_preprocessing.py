# -*- coding: utf-8 -*-
"""Project_Data_Preprocessing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12mDiMbPg2TL8lclQHeZX3Jt6pwg5RBRK
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
import torch.nn.functional as F
from tqdm.auto import tqdm
import copy
import matplotlib.pyplot as plt
import gdown
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, roc_auc_score

from google.colab import files

# Create a file upload dialog
uploaded_files = files.upload()

# Print the names of the uploaded files
for file_name in uploaded_files.keys():
    print('User uploaded file "{name}" with length {length} bytes'.format(
        name=file_name, length=len(uploaded_files[file_name])))

file_name = 'test_video_altered.mp4'

import shutil
import os
os.makedirs("test_video")
shutil.move(file_name ,"/content/test_video/"+file_name)

!pip3 install MTCNN

import cv2
import os
from tqdm import tqdm
from mtcnn.mtcnn import MTCNN
from PIL import Image
import numpy as np
import time

def video_to_frames(video_path, output_dir, subsample_rate=1):
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    # Get the video's frame rate
    frame_rate = int(video_capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break  # Break the loop if no more frames are available

        # Save frames at the specified subsample rate
        if frame_count % (frame_rate // subsample_rate) == 0:
            output_frame_path = os.path.join(output_dir, f"{os.path.basename(video_path)}_frame_{frame_count}.jpg")
            cv2.imwrite(output_frame_path, frame)

        frame_count += 1

    video_capture.release()
    print(f"{frame_count // (frame_rate // subsample_rate)} frames extracted from {os.path.basename(video_path)}")

# Define the path to the directory containing the altered videos
video_dir = "/content/test_video"

# Define the directory where you want to save the frames
output_dir = "/content/test_video_frames"

# Define the subsample rate (frames per second)
subsample_rate = 1

# Get list of all video files in the directory
video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
video_paths = [os.path.join(video_dir, file) for file in video_files]

# Extract frames from each video
for video_path in tqdm(video_paths):
    video_to_frames(video_path, output_dir, subsample_rate)

def crop_and_resize(input_dir, output_dir, size=(224, 224)):
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize the MTCNN detector
    detector = MTCNN()

    # Loop through each image in the input directory
    for img_file in tqdm(os.listdir(input_dir)):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            img_path = os.path.join(input_dir, img_file)
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

            # Detect faces in the image
            faces = detector.detect_faces(np.array(img))

            # If a face is detected, crop the face, resize it, and save it
            if faces:
                x, y, width, height = faces[0]['box']
                face = img.crop((x, y, x + width, y + height))
                face = face.resize(size)
                face.save(os.path.join(output_dir, img_file))

input_dir = '/content/test_video_frames'
output_dir = '/content/test_video_frames_cropped/all_images'

crop_and_resize(input_dir, output_dir)

transform = transforms.Compose([
    transforms.Resize((224, 224)), # Resize to the input size of the model
    transforms.ToTensor(),
])

test_dataset = datasets.ImageFolder(root='/content/test_video_frames_cropped', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def check_data_loader_dim(loader):
    # Checking the dataset
    for images, labels in loader:
        print('Image batch dimensions:', images.shape)
        print('Image label dimensions:', labels.shape)
        break

check_data_loader_dim(test_loader)

# Load the pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)

# Freeze model parameters
for param in model.parameters():
    param.requires_grad = False

# Replace the last fully connected layer (fc) for binary classification
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# Load the entire model
trained_model = torch.load('Resnet18_model.pth')

# Set to evaluation mode for inference
trained_model.eval()

# Move model to the right device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trained_model.to(device)

!pip install torchsummary

from torchsummary import summary

summary(trained_model, (3, 224, 224))

class_names = ['altered_frames_cropped', 'original_frames_cropped']

from collections import Counter

predictions = []
with torch.no_grad():
    for data, _ in test_loader:
        # Move data to the same device as the model
        data = data.to(device)

        # Forward pass
        output = trained_model(data)

        # Apply sigmoid if your model doesn't include it as the final activation
        # output = torch.sigmoid(output)

        # Convert output probabilities to class predictions (0 or 1)
        predicted_classes = (output > 0.0).int()
        predictions.extend(predicted_classes.cpu().numpy())

# Flatten the list of predictions
flat_predictions = [item for sublist in predictions for item in sublist]
prediction_counts = Counter(flat_predictions)
print(prediction_counts)
most_common_prediction = prediction_counts.most_common(1)[0][0]

if most_common_prediction == 0:
  print('Predicted Class: Altered Image' )
else:
  print('Predicted Class: Original Image' )

!pip install torchcam

import torch
import random
from torchvision import transforms
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
import matplotlib.pyplot as plt

# Load your trained model
model = torch.load('Resnet18_model.pth')
model.eval()

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Enable gradients for the target layer
target_layer = 'layer4'  # Specify the target layer for Grad-CAM
for param in getattr(model, target_layer).parameters():
    param.requires_grad = True

# Preprocessing function
preprocess = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Instantiate your CAM extractor
cam_extractor = GradCAM(model, target_layer)

# Randomly sample 5 images from the test_loader
random_indices = random.sample(range(len(test_loader.dataset)), 5)
random_images, random_labels = zip(*[(test_loader.dataset[i][0], test_loader.dataset[i][1]) for i in random_indices])

# Process and visualize each random image
for i, (image, label) in enumerate(zip(random_images, random_labels)):
    # Preprocess the image
    input_tensor = preprocess(image).to(device).unsqueeze(0)  # Add batch dimension

    # Calculate the CAM
    with torch.set_grad_enabled(True):
        out = model(input_tensor)
        activation_map = cam_extractor(out.squeeze().argmax().item(), out)

    # Check if the activation_map is a list and take the first element if so
    if isinstance(activation_map, list):
        activation_map = activation_map[0]

    # Move the activation map to CPU and convert to numpy
    activation_map_np = activation_map.squeeze().cpu().numpy()

    # Convert the activation map to PIL image
    activation_map_pil = transforms.ToPILImage()(activation_map_np)

    # Resize the CAM and overlay it
    result = overlay_mask(transforms.ToPILImage()(image), activation_map_pil, alpha=0.5)

    # Displaying the original image with CAM
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(transforms.ToPILImage()(image))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.title('Image with CAM Overlay')
    plt.axis('off')

    plt.show()

# Turn off gradients for the target layer
for param in getattr(model, target_layer).parameters():
    param.requires_grad = False
