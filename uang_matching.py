import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import cv2
import numpy as np

class ImageProcessor(VideoProcessorBase):
    def __init__(self):
        self._capture_image = False
        self._image_data = None

    def recv(self, frame):
        if self._capture_image:
            # Convert frame to BGR format
            img = frame.to_ndarray(format="bgr24")
            # Save the captured image data
            self._image_data = img
            # Reset capture flag
            self._capture_image = False
        return frame

def main():
    st.title("Capture Image from Camera")

    # Initialize video processor
    processor = ImageProcessor()

    # Display video stream from camera
    webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, video_processor_factory=processor)

    # Button to capture image
    if st.button("Capture Image"):
        processor._capture_image = True
        st.success("Image captured!")

        # Display captured image
        if processor._image_data is not None:
            st.image(processor._image_data, channels="BGR")

if __name__ == "__main__":
    main()
