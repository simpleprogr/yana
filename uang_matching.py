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

def get_currency_color(hue_value):
    sound_folder = os.path.join(os.path.dirname(__file__), 'sound')
    if hue_value < 0:
        return "MATA UANG"
    elif hue_value < 10:
        playsound(os.path.join(sound_folder, '5000.mp3'))
        return "Nominal Uang 5000"
    elif hue_value < 30:
        playsound(os.path.join(sound_folder, '1000.mp3'))
        return "Nominal Uang 1000"
    elif hue_value < 75:
        playsound(os.path.join(sound_folder, '20000.mp3'))
        return "Nominal Uang 20.000"
    elif hue_value < 102:
        playsound(os.path.join(sound_folder, '2000.mp3'))
        return "Nominal Uang 2000"
    elif hue_value < 105:
        playsound(os.path.join(sound_folder, '50000.mp3'))
        return "Nominal Uang 50.000"
    elif hue_value < 160:
        playsound(os.path.join(sound_folder, '10000.mp3'))
        return "Nominal Uang 10.000"
    elif hue_value < 177:
        playsound(os.path.join(sound_folder, '100000.mp3'))
        return "Nominal Uang 100.000"
    else:
        return "MATA UANG"

# Function to start camera
def start_camera():
    st.session_state.camera_active = True
    st.session_state.currency_detected = False

# Function to stop camera
def stop_camera():
    st.session_state.camera_active = False

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
    for template in template_data:
        (tmp_height, tmp_width) = template['glob'].shape[:2]
        image_test_p = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_test_p = cv2.Canny(image_test_p, 50, 200)
        found = None
        thershold = 0.4
        for scale in np.linspace(0.2, 1.0, 20)[::-1]: 
            resized = imutils.resize(image_test_p, width=int(image_test_p.shape[1] * scale))
            r = image_test_p.shape[1] / float(resized.shape[1]) 
            if resized.shape[0] < tmp_height or resized.shape[1] < tmp_width:
                break

            result = cv2.matchTemplate(resized, template['glob'], cv2.TM_CCOEFF_NORMED)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)
                if maxVal >= thershold: 
                    hasil = f"Template : {template['nominal']} dideteksi"
        if found is not None: 
            (maxVal, maxLoc, r) = found
            (startX, startY) = (int(maxLoc[0]*r), int(maxLoc[1] * r))
            (endX, endY) = (int((maxLoc[0] + tmp_width) * r), int((maxLoc[1] + tmp_height) * r))
            if maxVal >= thershold:
                cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
                playsound_mapping(int(template['nominal']))

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
            /* background-color: #F0F0F0; Warna latar belakang */
            /* padding: 10px; Padding di sekitar teks */
            /* border-radius: 5px; Border radius */
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
        start_button = st.button("Start Camera", on_click=start_camera, use_container_width=True)
        stop_button = st.button("Stop Camera", on_click=stop_camera, use_container_width=True)
        capture_button = st.button("Capture Image", use_container_width=True)

        stframe = st.empty()

    # Initialize camera
        cap = cv2.VideoCapture(0)

    # Set resolution
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1100)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)

    while st.session_state.camera_active:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image.")
            break

        stframe.image(frame, channels="BGR")

        if capture_button:
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            height, width, _ = frame.shape

            cx = int(width / 2)
            cy = int(height / 2)

            pixel_center = hsv_frame[cy, cx]
            hue_value = pixel_center[0]

            color = get_currency_color(hue_value)

            pixel_center_bgr = frame[cy, cx]
            b, g, r = int(pixel_center_bgr[0]), int(pixel_center_bgr[1]), int(pixel_center_bgr[2])

            #cv2.rectangle(frame, (cx - 420, 120), (cx + 450, 20), (255, 255, 255), -1)
            #cv2.putText(frame, color, (cx - 300, 50), 0, 3, (b, g, r), 5)
            cv2.circle(frame, (cx, cy), 5, (25, 25, 25), 3)

            stframe.image(frame, channels="BGR")
            st.write(f"Hasil Deteksi : {color}")

            st.session_state.currency_detected = True
            st.session_state.camera_active = False

    if not st.session_state.camera_active:
        cap.release()
        #cv2.destroyAllWindows()
    
    col1, col2, col3 = st.columns([1, 4, 1])

    with col2:
        uploaded_file = st.file_uploader("", type=["jpg", "png"])
    
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_test = cv2.imdecode(file_bytes, 1)
            
            detect(image_test)
            
            st.image(image_test, channels="BGR")
            
            st.write(hasil)

    # Menggunakan CSS dari Streamlit untuk mengatur teks di tengah
    st.markdown("""
        <style>
        .foother {
            text-align: center;
            font-size: 16px; /* Ukuran font */
            color: #FF5733; /* Warna teks */
            background-color: #F0F0F0; /* Warna latar belakang */
            /* padding: 10px; Padding di sekitar teks */
            /* border-radius: 5px; Border radius */
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
            /* background-color: #F0F0F0; Warna latar belakang */
            /* padding: 10px; Padding di sekitar teks */
            /* border-radius: 5px; Border radius */
        }
        </style>
        """, unsafe_allow_html=True)

    # Memasukkan teks ke dalam div dengan kelas centered-text
    st.markdown('<div class="foother">Yana Wulandari</div><div class="foother">0701202073</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
