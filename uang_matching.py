import streamlit as st
import cv2
import numpy as np
import imutils
import os
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import av

_capture, image_test = False, None
template_data = []
hasil = ''
audio_file = ''

# Initialize session state for camera control
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'currency_detected' not in st.session_state:
    st.session_state.currency_detected = False

def get_currency_color(hue_value):
    if hue_value < 0:
        return "MATA UANG"
    elif hue_value < 10:
        return "Nominal Uang 5000", 'sound/5000.mp3'
    elif hue_value < 30:
        return "Nominal Uang 1000", 'sound/1000.mp3'
    elif hue_value < 75:
        return "Nominal Uang 20000", 'sound/20000.mp3'
    elif hue_value < 102:
        return "Nominal Uang 2000", 'sound/2000.mp3'
    elif hue_value < 105:
        return "Nominal Uang 50000", 'sound/50000.mp3'
    elif hue_value < 160:
        return "Nominal Uang 10000", 'sound/10000.mp3'
    elif hue_value < 177:
        return "Nominal Uang 100000", 'sound/100000.mp3'
    else:
        return "MATA UANG", None

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
        nominal = os.path.basename(template_file).replace('.jpg', '')
        template_data.append({"glob": tmp, "nominal": nominal})

def detect(img):     
    global template_data, hasil, audio_file
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
        audio_file = playsound_mapping(best_match['nominal'])

def playsound_mapping(nominal):
    try:
        nominal = int(nominal)
    except ValueError:
        return None

    if 0 <= nominal <= 9:
        return 'sound/1000.mp3'
    elif 8 <= nominal <= 21:
        return 'sound/2000.mp3'
    elif 20 <= nominal <= 34:
        return 'sound/5000.mp3'
    elif 33 <= nominal <= 48:
        return 'sound/10000.mp3'
    elif 47 <= nominal <= 57:
        return 'sound/20000.mp3'
    elif 56 <= nominal <= 69:
        return 'sound/50000.mp3'
    elif 68 <= nominal <= 84:
        return 'sound/100000.mp3'
    return None

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.template_data = []
        self.frame = None
        uang_matching()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame = img
        detect(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def save_image(self):
        if self.frame is not None:
            cv2.imwrite("captured_image.jpg", self.frame)

def main():
    st.set_page_config(page_title="Deteksi Nominal Mata Uang Menggunakan Template Matching", layout="centered")

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

    st.markdown("""
        <style>
        .title {
            text-align: center;
            font-size: 24px;
            color: #FF5733;
            margin-bottom: 36px;
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown('<div class="title">DETEKSI NOMINAL MATA UANG MENGGUNAKAN TEMPLATE MATCHING</div>', unsafe_allow_html=True)    
    
    uang_matching()

    st.write("---")

    webrtc_ctx = webrtc_streamer(
        key="example", 
        mode=WebRtcMode.SENDRECV, 
        video_processor_factory=VideoProcessor, 
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if st.button("Capture"):
        if webrtc_ctx.state.playing:
            webrtc_ctx.video_processor.save_image()
        
        if webrtc_ctx.video_processor.frame is not None:
            image_test = webrtc_ctx.video_processor.frame
            detect(image_test)
            
            st.image(image_test, channels="BGR")
            
            st.write(hasil)
            if audio_file:
                st.audio(audio_file, autoplay=True)

    st.markdown("""
        <style>
        .foother {
            text-align: center;
            font-size: 16px;
            color: #FF5733;
            background-color: #F0F0F0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <style>
        .foother2 {
            text-align: center;
            font-size: 16px;
            color: #FF5733;
            margin-top: 36px;
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown('<div class="foother">Yana Wulandari</div><div class="foother">0701202073</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
