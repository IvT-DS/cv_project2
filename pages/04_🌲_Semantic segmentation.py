# Contents of ~/my_app/pages/page_3.py
import streamlit as st
import PIL
import cv2
import numpy as np
import requests
import time

from ultralytics import YOLO
from io import BytesIO
from requests.models import MissingSchema
from PIL import Image

st.markdown("## Семантическая сегментация леса на аэрокосмических снимках 🌲")
st.sidebar.markdown("# Semantic segmentation page -->>")
st.sidebar.markdown("## Используй навигацию между страницами выше ⬆️")
