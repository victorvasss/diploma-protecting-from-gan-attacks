
import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

# Paths to the original and augmented image folders
original_folder = '/Users/victorvasss/Documents/polytech/diploma/research/dataset/original'
augmented_folder = '/Users/victorvasss/Documents/polytech/diploma/research/dataset/augmented'

# GAN dataset root
gan_root = '/Users/victorvasss/Documents/polytech/diploma/research/dataset'

# Lists to hold image paths and labels
image_paths = []
labels = []

# Add original images
for img_name in os.listdir(original_folder):
    image_paths.append(os.path.join(original_folder, img_name))
    labels.append(0)

# Add ESRGAN-augmented images
for img_name in os.listdir(augmented_folder):
    image_paths.append(os.path.join(augmented_folder, img_name))
    labels.append(1)

# Add GANGen-Detection subfolders
gan_models = ['AttGAN', 'BEGAN', 'CramerGAN', 'MMDGAN', 'S3GAN', 'SNGAN', 'STGAN']
for model in gan_models:
    real_dir = os.path.join(gan_root, model, '0_real')
    fake_dir = os.path.join(gan_root, model, '1_fake')
    if os.path.isdir(real_dir):
        for img_name in os.listdir(real_dir):
            image_paths.append(os.path.join(real_dir, img_name))
            labels.append(0)
    if os.path.isdir(fake_dir):
        for img_name in os.listdir(fake_dir):
            image_paths.append(os.path.join(fake_dir, img_name))
            labels.append(1)

# Combine into DataFrame
data = pd.DataFrame({'image_path': image_paths, 'label': labels})

# Split into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=106, stratify=data['label'])

# Save to pickle files
train_file = 'train_dataset.pkl'
val_file = 'val_dataset.pkl'

with open(train_file, 'wb') as f:
    pickle.dump(train_data, f)

with open(val_file, 'wb') as f:
    pickle.dump(val_data, f)

print(f"Training dataset saved to {train_file}")
print(f"Validation dataset saved to {val_file}")
print(f"Total samples: {len(data)}, Train: {len(train_data)}, Val: {len(val_data)}")
