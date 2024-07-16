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
            img = frame.to_ndarray(format="bgr24")
            self._image_data = img
            self._capture_image = False
        return frame

def main():
    st.title("Capture Image from Camera")

    # Initialize video processor
    processor = ImageProcessor()

    # Set media stream constraints to select rear camera
    media_stream_constraints = {
        "video": {
            "facingMode": "environment"  # Use "environment" for rear camera
        }
    }

    # Display video stream from camera
    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=ImageProcessor,
        media_stream_constraints=media_stream_constraints
    )

    # Button to capture image
    if st.button("Capture Image"):
        processor._capture_image = True
        st.success("Image captured!")

        # Display captured image
        if processor._image_data is not None:
            st.image(processor._image_data, channels="BGR")

if __name__ == "__main__":
    main()
