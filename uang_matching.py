import streamlit as st
import glob
import cv2
import numpy as np
import imutils
from playsound import playsound
import os

_capture, image_test = False, None
template_data = []
hasil = ''

# Initialize session state for camera control
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'currency_detected' not in st.session_state:
    st.session_state.currency_detected = False

def detect_by_color(img):
    hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    height, width, _ = img.shape

    cx = int(width / 2)
    cy = int(height / 2)

    pixel_center = hsv_frame[cy, cx]
    hue_value = pixel_center[0]

    if hue_value < 10 or hue_value > 160:
        playsound('sound/10000.mp3')
        return "Nominal Uang 10000"
    elif 10 <= hue_value < 30:
        playsound('sound/1000.mp3')
        return "Nominal Uang 1000"
    elif 30 <= hue_value < 50:
        playsound('sound/2000.mp3')
        return "Nominal Uang 2000"
    elif 50 <= hue_value < 70:
        playsound('sound/5000.mp3')
        return "Nominal Uang 5000"
    elif 70 <= hue_value < 90:
        playsound('sound/20000.mp3')
        return "Nominal Uang 20000"
    elif 90 <= hue_value < 110:
        playsound('sound/50000.mp3')
        return "Nominal Uang 50000"
    elif 110 <= hue_value < 130:
        playsound('sound/100000.mp3')
        return "Nominal Uang 100000"
    else:
        return "Tidak Teridentifikasi"

def uang_matching():
    global template_data
    template_files = glob.glob('template/*.jpg', recursive=True)
    st.write("Template loaded:", template_files)
    
    for template_file in template_files:
        tmp = cv2.imread(template_file)
        tmp = imutils.resize(tmp, width=int(tmp.shape[1]*0.5))  
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)  
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        tmp = cv2.filter2D(tmp, -1, kernel)
        tmp = cv2.blur(tmp, (3, 3)) 
        tmp = cv2.Canny(tmp, 50, 200)
        nominal = template_file.replace('template\\', '').replace('.jpg', '')
        template_data.append({"glob": tmp, "nominal": nominal})

def detect(img):     
    global template_data, hasil
    best_match = {"value": 0, "nominal": None, "location": None, "scale": 1}
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(img_gray, 50, 200)
    
    for template in template_data:
        tmp_height, tmp_width = template['glob'].shape[:2]
        for scale in np.linspace(0.2, 1.0, 20)[::-1]: 
            resized = imutils.resize(img_canny, width=int(img_canny.shape[1] * scale))
            r = img_canny.shape[1] / float(resized.shape[1]) 
            if resized.shape[0] < tmp_height or resized.shape[1] < tmp_width:
                break

            result = cv2.matchTemplate(resized, template['glob'], cv2.TM_CCOEFF_NORMED)
            _, maxVal, _, maxLoc = cv2.minMaxLoc(result)
            if maxVal > best_match["value"]:
                best_match.update({"value": maxVal, "nominal": template['nominal'], "location": maxLoc, "scale": r})
    
    if best_match["value"] >= 0.4:
        startX, startY = int(best_match["location"][0] * best_match["scale"]), int(best_match["location"][1] * best_match["scale"])
        endX, endY = int((best_match["location"][0] + tmp_width) * best_match["scale"]), int((best_match["location"][1] + tmp_height) * best_match["scale"])
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
        hasil = f"Template : {best_match['nominal']} dideteksi"
        playsound_mapping(int(best_match['nominal']))
    else:
        hasil = "Tidak dapat mendeteksi nominal dengan template matching"

def playsound_mapping(nominal):
    sound_folder = os.path.join(os.path.dirname(__file__), 'sound')
    if 0 <= nominal <= 9:
        playsound(os.path.join(sound_folder, '1000.mp3'))
    elif 8 <= nominal <= 21:
        playsound(os.path.join(sound_folder, '2000.mp3'))
    elif 20 <= nominal <= 34:
        playsound(os.path.join(sound_folder, '5000.mp3'))
    elif 33 <= nominal <= 48:
        playsound(os.path.join(sound_folder, '10000.mp3'))
    elif 47 <= nominal <= 57:
        playsound(os.path.join(sound_folder, '20000.mp3'))
    elif 56 <= nominal <= 69:
        playsound(os.path.join(sound_folder, '50000.mp3'))
    elif 68 <= nominal <= 84:
        playsound(os.path.join(sound_folder, '100000.mp3'))

def main():
    st.set_page_config(page_title="Deteksi Nominal Mata Uang Menggunakan Template Matching", layout="centered")

    # Inject custom CSS for background color
    st.markdown(
        """
        <style>
        body {
            background-color: #fa8072;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Menggunakan CSS dari Streamlit untuk mengatur teks di tengah
    st.markdown("""
        <style>
        .title {
            text-align: center;
            font-size: 24px; /* Ukuran font */
            color: #FF5733; /* Warna teks */
            margin-bottom: 36px;
        }
        </style>
        """, unsafe_allow_html=True)

    # Memasukkan teks ke dalam div dengan kelas centered-text
    st.markdown('<div class="title">DETEKSI NOMINAL MATA UANG MENGGUNAKAN TEMPLATE MATCHING</div>', unsafe_allow_html=True)    
    
    uang_matching()

    st.write("---")

    # Use columns to center the elements on the page
    col1, col2, col3 = st.columns([1, 4, 1])

    with col2:
        start_button = st.button("Start Camera", on_click=lambda: st.session_state.update(camera_active=True), use_container_width=True)
        stop_button = st.button("Stop Camera", on_click=lambda: st.session_state.update(camera_active=False), use_container_width=True)
        capture_button = st.button("Capture Image", use_container_width=True)

        stframe = st.empty()

        # Initialize camera
        camera_input = st.camera_input("Camera")


    if camera_input:

        img = camera_input.get_image()

        st.image(img, channels="BGR")


        # Detect the nominal value

        nominal_value = detect_by_color(img)

        st.write("Nominal Value:", nominal_value)
    
    col1, col2, col3 = st.columns([1, 4, 1])

    with col2:
        uploaded_file = st.file_uploader("", type=["jpg", "png"])
    
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_test = cv2.imdecode(file_bytes, 1)
            
            detect(image_test)
            
            st.image(image_test, channels="BGR")
            
            st.write(hasil)

    st.markdown("""
        <style>       

        .foother {
            text-align: center;
            font-size: 16px; /* Ukuran font */
            color: #FF5733; /* Warna teks */
            background-color: #F0F0F0; /* Warna latar belakang */
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <style>
        .foother2 {
            text-align: center;
            font-size: 16px; /* Ukuran font */
            color: #FF5733; /* Warna teks */
            margin-top: 36px;
        }
        </style>
        """, unsafe_allow_html=True)

    # Memasukkan teks ke dalam div dengan kelas centered-text
    st.markdown('<div class="foother">Yana Wulandari</div><div class="foother">0701202073</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
