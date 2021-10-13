import streamlit as st
from joblib import load
import numpy as np
import cv2
import pickle
import bitarray

from imageio import imread, imwrite
from PIL import Image
from bitstring import BitArray

from app_utils import *


model = "fdf128_rcnn512"

anonymizer, cfg = build_anonymizer(
    model, opts=None, config_path=None,
    return_cfg=True)

# import pickle
# # pickle.dump( anonymizer, open( "anonymizer.p", "wb" ) )

# file = open("anonymizer.p",'rb')
# object_file = pickle.load(file)
# file.close()

st.title("Image Steganography App")
st.subheader("Encode Information In Images")

uploaded_file = st.file_uploader("Upload Image")

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    anonymized_images, image_annotations = anonymizer.detect_and_anonymize_images(
        [image], None, return_annotations=True)
    st.image(image, caption="Input Image", channels="BGR")

    # ------------------------------------------------------------
    x0, y0, x1, y1 = image_annotations[0].bbox_XYXY[0]
    orig_cutout = image[y0:y1, x0:x1]  # [:, :, ::-1]

    # compression
    imwrite("orig_cutout_saved.jpg", orig_cutout)
    img = Image.open("orig_cutout_saved.jpg")
    img.save("COMPRESSED_orig_cutout_saved.jpg", optimize=True, quality=50)

    # extract bits to hide
    with open('COMPRESSED_orig_cutout_saved.jpg', "rb") as compressed_image:
        f = compressed_image.read()
        b = bytearray(f)
        target = BitArray(bytes=b).bin
        target = np.array([int(elem) for elem in list(target)])
        #target = torch.tensor([int(elem) for elem in list(target)])
    # ------------------------------------------------------------

    encoded_image = lsb_encoding(anonymized_images[0], x0, y0, x1, y1, target)
    message = lsb_decode(encoded_image, x0, y0, x1, y1)

    st.image(anonymized_images[0], caption="Anonymized Image", channels="BGR")

    st.image(encoded_image, caption="Encoded Anonymized Image", channels="BGR")

    bits = bitarray.bitarray(message)
    recovered_bytes = np.asarray(bytearray(bits), dtype=np.uint8)
    recovered_face = cv2.imdecode(recovered_bytes, 1)[:, :, ::-1]

    st.image(recovered_face, caption="Recovered Face", channels="BGR")

    recovered_image = add_face_to_image(
        recovered_face, anonymized_images[0], x0, y0, x1, y1)

    st.image(recovered_image, caption="Recovered Image", channels="BGR")
