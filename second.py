import streamlit as st
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
import os

@st.cache_data
def read_data():
    vec_all = np.load("vec_all.npy", allow_pickle=True)  # Allow pickle for object arrays
    name_all = np.load("name_all.npy", allow_pickle=True)  # Allow pickle for object arrays
    return vec_all, name_all

vecs, names = read_data()

# Ensure names array is not empty
if len(names) == 0:
    st.error("The 'names' array is empty. Please check the data.")
    st.stop()  # Stop the app if no names are available

_, fcol2, _ = st.columns(3)
scol1, scol2 = st.columns(2)

ch = scol1.button("Start/Change")
fs = scol2.button("Find Similar")

image_folder = "C:/Users/kgt/OneDrive/Desktop/coding/virtualintern/imagesearchengine"

# Initialize session_state if not already initialized
if "Disp_img" not in st.session_state:
    st.session_state["Disp_img"] = None

if ch:
    # Ensure names is not empty before selecting a random name
    if len(names) > 0:
        rand_name = names[np.random.randint(len(names))]
        fcol2.image(Image.open(os.path.join(image_folder, rand_name)))
        st.session_state["Disp_img"] = rand_name
        st.write(st.session_state["Disp_img"])

if fs and st.session_state["Disp_img"] is not None:
    c1, c2, c3, c4, c5 = st.columns(5)
    
    # Ensure that the index extraction works correctly
    idx = np.argwhere(names == st.session_state["Disp_img"])[0][0]
    target_vec = vecs[idx]
    
    fcol2.image(Image.open(os.path.join(image_folder, st.session_state["Disp_img"])))
    
    # Find the top 5 similar images
    top5 = cdist(target_vec[None, ...], vecs).squeeze().argsort()[1:6]
    
    c1.image(Image.open(os.path.join(image_folder, names[top5[0]])))
    c2.image(Image.open(os.path.join(image_folder, names[top5[1]])))
    c3.image(Image.open(os.path.join(image_folder, names[top5[2]])))
    c4.image(Image.open(os.path.join(image_folder, names[top5[3]])))
    c5.image(Image.open(os.path.join(image_folder, names[top5[4]])))
