import streamlit as st
import cv2
import numpy as np

def main():
    st.title("Take and Display Picture with BGR Channel")

    # Capture image from camera input
    picture = st.camera_input("Take a picture")

    if picture:
        # Convert the image to BGR format
        frame_bgr = cv2.cvtColor(np.array(picture), cv2.COLOR_RGB2BGR)

        # Display the image with BGR channel
        st.image(frame_bgr, channels="BGR", use_column_width=True)

if __name__ == "__main__":
    main()
