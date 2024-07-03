import streamlit as st
import glob
import cv2
import numpy as np
import imutils
from PIL import Image
from io import BytesIO
from playsound import playsound
import tempfile

_capture, image_test = False, None
template_data = []
hasil = ''

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
    if 0 <= nominal <= 9:
        playsound('E:/New Skripsi/2024/Yana/Skripsi Template Matching/sound/1000.mp3')
    elif 8 <= nominal <= 21:
        playsound('E:/New Skripsi/2024/Yana/Skripsi Template Matching/sound/2000.mp3')
    elif 20 <= nominal <= 34:
        playsound('E:/New Skripsi/2024/Yana/Skripsi Template Matching/sound/5000.mp3')
    elif 33 <= nominal <= 48:
        playsound('E:/New Skripsi/2024/Yana/Skripsi Template Matching/sound/10000.mp3')
    elif 47 <= nominal <= 57:
        playsound('E:/New Skripsi/2024/Yana/Skripsi Template Matching/sound/20000.mp3')
    elif 56 <= nominal <= 69:
        playsound('E:/New Skripsi/2024/Yana/Skripsi Template Matching/sound/50000.mp3')
    elif 68 <= nominal <= 84:
        playsound('E:/New Skripsi/2024/Yana/Skripsi Template Matching/sound/100000.mp3')

def main():
    st.set_page_config(page_title="Deteksi Nominal Mata Uang Menggunakan Template Matching", layout="centered")

    st.title("Deteksi Nominal Mata Uang Menggunakan Template Matching")
    st.subheader("Yana Wulandari")
    st.subheader("0701202073")
    
    uang_matching()

    st.write("---")

    picture = st.camera_input("AMBIL FOTO")
    if picture:
        file_bytes = np.asarray(bytearray(picture), dtype=np.uint8)
        image_test = cv2.imdecode(file_bytes, 1)
        detect(image_test)
        #st.image(picture)

    if st.button('BUKA KAMERA'):
        st.warning("Fungsi kamera belum diimplementasikan dalam versi Streamlit.")
    
    uploaded_file = st.file_uploader("BUKA PENYIMPANAN", type=["jpg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_test = cv2.imdecode(file_bytes, 1)
        
        detect(image_test)
        
        st.image(image_test, channels="BGR")
        
        st.write(hasil)

if __name__ == "__main__":
    main()