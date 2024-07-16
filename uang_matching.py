import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import av

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.cx = 0
        self.cy = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        height, width, _ = img.shape
        
        self.cx = int(width / 2)
        self.cy = int(height / 2)
        
        pixel_center = hsv_frame[self.cy, self.cx]
        hue_value = pixel_center[0]

        color = "Undefined"
        if hue_value < 0:
            color = "MATA UANG"
        elif hue_value < 10:
            color = "5000"
        elif hue_value < 30:
            color = "1000"
        elif hue_value < 75:
            color = "20.000"
        elif hue_value < 102:
            color = "2000"
        elif hue_value < 105:
            color = "50.000"
        elif hue_value < 160:
            color = "10.000"
        elif hue_value < 177:
            color = "100.000"
        else:
            color = "MATA UANG"
        
        pixel_center_bgr = img[self.cy, self.cx]
        b, g, r = int(pixel_center_bgr[0]), int(pixel_center_bgr[1]), int(pixel_center_bgr[2])  

        cv2.rectangle(img, (self.cx - 420, 120), (self.cx + 450, 20), (255, 255, 255), -1)
        cv2.putText(img, color, (self.cx - 300, 50), 0, 3, (b, g, r), 5)
        cv2.circle(img, (self.cx, self.cy), 5, (25, 25, 25), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.title("Deteksi Nominal Mata Uang")

    webrtc_ctx = webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, video_processor_factory=VideoProcessor)

    if webrtc_ctx.video_processor:
        st.write("Webcam is active. Please show the currency note to the camera.")

if __name__ == "__main__":
    main()
