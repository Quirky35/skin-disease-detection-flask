import pandas as pd
import os
from sklearn.model_selection import train_test_split
import shutil

def prepare_dataset(csv_path, img_dir, output_dir):
    df = pd.read_csv(csv_path)

    label_map = {
        'nv': 'melanocytic_nevi',
        'mel': 'melanoma',
        'bkl': 'benign_keratosis',
        'bcc': 'basal_cell_carcinoma',
        'akiec': 'actinic_keratoses',
        'vasc': 'vascular_lesions',
        'df': 'dermatofibroma'
    }

    df['label'] = df['dx'].map(label_map)

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    for subset, data in [('train', train_df), ('test', test_df)]:
        for _, row in data.iterrows():
            label = row['label']
            img_id = row['image_id'] + '.jpg'
            src = os.path.join(img_dir, img_id)
            dst_dir = os.path.join(output_dir, subset, label)
            os.makedirs(dst_dir, exist_ok=True)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(dst_dir, img_id))

    print("✅ Dataset organized into train/test folders.")

# Run this when script is executed directly
if __name__ == "__main__":
    prepare_dataset(
        csv_path="dataset/HAM10000_metadata.csv",
        img_dir="dataset/HAM10000_images",
        output_dir="dataset"
    )
