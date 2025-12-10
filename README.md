📌 Skin Disease Detection Using Deep Learning (Flask Web App)

An AI-powered web application that detects 7+ types of skin diseases from images using a trained deep-learning model (MobileNetV2).
The application provides:

🧠 AI-based disease prediction

🧑‍⚕️ Doctor recommendation

💊 Medicine suggestions

🗺️ Nearby clinic map using OpenStreetMap (Overpass API)

📄 Downloadable health report

🚀 Project Features
🔹 AI Skin Disease Classification

Trained on the HAM10000 Kaggle Dataset, the model predicts:

Actinic Keratoses

Basal Cell Carcinoma

Benign Keratosis

Dermatofibroma

Melanocytic Nevi

Melanoma

Vascular Lesions

Not Skin Disease (custom 8th class)

🔹 Doctor Recommendation

Displays the best specialist with hospital info + Google Maps link.

🔹 Medicine Suggestions

Provides general over-the-counter medicine suggestions.

🔹 Nearby Clinics on Map

Using OpenStreetMap + OverpassAPI + Folium, finds hospitals/clinics near the user.

🔹 Downloadable AI Report

Generates a .txt report containing prediction + medicines + doctor info.

🔹 Beautiful UI

Modern, responsive UI built using HTML + CSS with a medical theme.

🧠 Technologies Used
Component	Technology
ML Framework	TensorFlow / Keras
Model Architecture	MobileNetV2
Web Framework	Flask
Frontend	HTML, CSS, JS
Maps API	OpenStreetMap (OSM), Overpass API, Folium
Dataset	HAM10000 (Kaggle)
Deployment	Localhost / GitHub project
📥 Dataset: HAM10000 (Kaggle)

Download from Kaggle:

🔗 https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

After downloading, you will get:
Image files
Two metadata CSV files

🛠️ How to Build This Project From Scratch
1️⃣ Step 1 — Download Dataset from Kaggle

Go to Kaggle link above
Download the ZIP
Extract it
You will get folders like:

HAM10000_images_part_1/
HAM10000_images_part_2/
HAM10000_metadata.csv

2️⃣ Step 2 — Organize Dataset

You need to re-arrange images into class-wise folders.
A common structure is:

dataset/
 ├── actinic_keratoses/
 ├── basal_cell_carcinoma/
 ├── benign_keratosis/
 ├── dermatofibroma/
 ├── melanocytic_nevi/
 ├── melanoma/
 ├── vascular_lesions/
 └── not_skin_disease/    ← custom added


You can write a Python script to automatically move images based on metadata CSV.

3️⃣ Step 3 — Train the Model
🧪 Model Training Script (Example)
MobileNetV2 fine-tuning:
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # freeze layers
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(7, activation='softmax')  # 7 classes
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_ds, validation_data=val_ds, epochs=10)
model.save("skin_disease_model.h5")

You saved this model in your project and used it for prediction.

🧩 4️⃣ Step 4 — Build Flask App
Directory structure:
flask_app/
 ├── app.py
 ├── skin_disease_model.h5
 ├── templates/
 │     ├── index.html
 │     └── result.html
 ├── static/
 │     ├── clinic_map.html
 │     └── style.css
 ├── uploads/
 └── src/

5️⃣ Step 5 — Run the Project
Install dependencies:
pip install tensorflow flask pillow numpy folium requests
Run the Flask app:
python app.py

Open browser:
👉 http://127.0.0.1:5000/
Folder Structure (Final Repo)
skin-disease-detection-flask/
│
├── app.py
├── skin_disease_model.h5
├── README.md
├── .gitignore
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── static/
│   ├── clinic_map.html
│   ├── hospital_map.html
│   └── style.css
│
├── uploads/ (ignored)
└── src/

🚀 Future Enhancements

Deploy on Render / Railway / AWS
Add user login system
Improve accuracy with Vision Transformers
Add image preprocessing (hair removal, denoising)
Add severity detection
Add medical PDF report generation

🙏 Acknowledgements

Kaggle HAM10000 dataset
OpenStreetMap (OSM)
TensorFlow
Flask

📸 Screenshots
Home Page
<img width="916" height="869" alt="Screenshot 2025-08-25 231058" src="https://github.com/user-attachments/assets/83b7a67e-f64b-46eb-be66-729fea8ee7da" />



Prediction Result Page

<img width="1113" height="810" alt="Screenshot 2025-08-25 231201" src="https://github.com/user-attachments/assets/645d267b-578a-4189-b02b-f20e483f92a3" />


Map Page

<img width="929" height="874" alt="Screenshot 2025-08-25 231254" src="https://github.com/user-attachments/assets/26711b96-508e-451a-a479-174904298c70" />
