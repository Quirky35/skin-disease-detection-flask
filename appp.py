import os
import numpy as np
import requests
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import folium

app = Flask(__name__)
MODEL_PATH = 'skin_disease_model.h5'
model = load_model(MODEL_PATH)

# Class labels
class_labels = [
    "Actinic keratoses", "Basal cell carcinoma", "Benign keratosis-like lesions",
    "Dermatofibroma", "Melanocytic nevi", "Melanoma","Not Skin Disease", "Vascular lesions"
]

# Doctor info
doctor_info = {
    "Actinic keratoses": '<a href="https://maps.google.com/?q=LeJeune+Skin+Clinic+Bangalore" target="_blank">Dr. Shuba Dharmana, LeJeune Skin Clinic, Bangalore – +91-80-4160-2592</a>',
    "Basal cell carcinoma": '<a href="https://maps.google.com/?q=Fortis+Cancer+Institute+Bangalore" target="_blank">Dr. Niti Raizada, Fortis Cancer Institute, Bangalore – +91-80-6621-4444</a>',
    "Benign keratosis-like lesions": '<a href="https://maps.google.com/?q=Dr+Dixit+Cosmetic+Dermatology+Bangalore" target="_blank">Dr. Rasya Dixit, Dixit Cosmetic Dermatology, Bangalore – +91-99018-90588</a>',
    "Dermatofibroma": '<a href="https://maps.google.com/?q=Cutis+Skin+Clinic+Bangalore" target="_blank">Dr. Sachith Abraham, Cutis Skin Clinic, Bangalore – +91-80-2222-0232</a>',
    "Melanocytic nevi": '<a href="https://maps.google.com/?q=Skinology+Centre+Bangalore" target="_blank">Dr. Sunaina Hameed, Skinology Centre, Bangalore – +91-96324-75534</a>',
    "Melanoma": '<a href="https://maps.google.com/?q=Cytecare+Cancer+Hospital+Bangalore" target="_blank">Dr. Belliappa, Cytecare Cancer Hospital, Bangalore – +91-80-2218-2200</a>',
    "Vascular lesions": '<a href="https://maps.google.com/?q=Manipal+Hospitals+Old+Airport+Road+Bangalore" target="_blank">Dr. Sangeeta A, Manipal Hospitals, Bangalore – +91-80-2222-1111</a>',
    "Not Skin Disease": "No doctor needed. No skin disease detected."
}


# Medicine suggestions
medicines = {
    "Actinic keratoses": ['5-Fluorouracil cream', 'Imiquimod'],
    "Basal cell carcinoma": ['Vismodegib', 'Surgical removal'],
    "Benign keratosis-like lesions": ['Moisturizers', 'Topical corticosteroids'],
    "Dermatofibroma": ['No treatment needed unless symptomatic'],
    "Melanocytic nevi": ['Regular monitoring'],
    "Melanoma": ['Immunotherapy', 'Surgical excision'],
    "Vascular lesions": ['Laser therapy', 'Beta-blockers'],
    "Not Skin Disease": ['No medicine needed']
}
# Disease descriptions
disease_descriptions = {
    "Actinic keratoses": "A rough, scaly patch on the skin caused by years of sun exposure. It is considered precancerous and can sometimes progress to skin cancer.",
    "Basal cell carcinoma": "A common form of skin cancer that begins in the basal cells, often appearing as a transparent bump on sun-exposed areas.",
    "Benign keratosis-like lesions": "Non-cancerous skin growths that may appear wart-like or scaly. They are harmless and often related to aging or sun exposure.",
    "Dermatofibroma": "A small, firm, harmless bump under the skin, usually caused by a reaction to a minor injury or insect bite.",
    "Melanocytic nevi": "Commonly known as moles — benign clusters of pigment cells. They are generally harmless but should be monitored for changes.",
    "Melanoma": "A serious type of skin cancer that develops in pigment cells. Early detection and treatment are crucial to prevent spreading.",
    "Vascular lesions": "Abnormal clusters of blood vessels appearing as red or purple marks on the skin, often treatable with laser therapy.",
    "Not Skin Disease": "The uploaded image does not indicate any skin disease. No medical concern detected."
}


# Get user location using IP
def get_location():
    # Fixed location: your exact coordinates
    return 12.872858348203009, 77.5758321938258


# Get nearby hospitals using Overpass API
def get_nearby_hospitals(lat, lon):
    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    (
      node["amenity"="clinic"](around:3000,{lat},{lon});
      node["amenity"="hospital"](around:3000,{lat},{lon});
    );
    out;
    """
    response = requests.post(overpass_url, data={'data': query})
    data = response.json()

    hospital_list = []
    for element in data['elements']:
        tags = element.get('tags', {})
        name = tags.get('name', 'Unnamed')
        address = tags.get('addr:full') or f"{tags.get('addr:street', '')}, {tags.get('addr:city', '')}"
        phone = tags.get('phone', 'N/A')
        hours = tags.get('opening_hours', 'N/A')
        lat = element['lat']
        lon = element['lon']

        hospital_list.append({
            'name': name,
            'address': address,
            'phone': phone,
            'hours': hours,
            'lat': lat,
            'lon': lon
        })
    return hospital_list

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict class
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    # Get location
    lat, lon = get_location()
    hospitals = get_nearby_hospitals(lat, lon)

    # Generate map
    clinic_map = folium.Map(location=[lat, lon], zoom_start=13)
    folium.Marker([lat, lon], tooltip='Your Location', icon=folium.Icon(color='blue')).add_to(clinic_map)

    for hospital in hospitals:
        name = hospital['name']
        address = hospital['address']
        phone = hospital['phone']
        hours = hospital['hours']
        h_lat = hospital['lat']
        h_lon = hospital['lon']

        maps_url = f"https://www.google.com/maps/search/?api=1&query={h_lat},{h_lon}"
        popup_html = f"""
        <b>{name}</b><br>
        📍 {address}<br>
        📞 {phone}<br>
        🕒 {hours}<br>
        <a href="{maps_url}" target="_blank">🗺️ Open in Google Maps</a>
        """

        folium.Marker(
            [h_lat, h_lon],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color='red', icon='plus-sign')
        ).add_to(clinic_map)

    map_path = os.path.join('static', 'clinic_map.html')
    clinic_map.save(map_path)
    report_filename = f"{file.filename}_report.txt"
    report_path = os.path.join('uploads', report_filename)
    with open(report_path, 'w') as f:
        f.write(f"Prediction: {predicted_class}\n")
        f.write(f"Recommended Doctor: {doctor_info[predicted_class]}\n")
        f.write("Suggested Medicines:\n")
        for med in medicines[predicted_class]:
            f.write(f" - {med}\n")

    

    return render_template(
    'result.html',
    prediction=predicted_class,
    doctor=doctor_info[predicted_class],
    medicines=medicines[predicted_class],
    description=disease_descriptions[predicted_class],
    map_file='clinic_map.html',
    filename=file.filename
)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory('uploads', filename, as_attachment=True)


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
