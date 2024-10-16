import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
from glob import glob
import os
from train import *
from predict import *
#K3_AB1550RX.jpg, K3_AA1081C.JPG, K4_AA1648LT.JPG, K4_AB1536AW.JPG, 

with st.sidebar:
    selected = option_menu("Menu",["Data Plat","Pelatihan","Deteksi Plat"],
                           icons=['table','stars','bi bi-subtract', 'gear', 'bi bi-search'], menu_icon="cast",
                           default_index=0, styles={
        "container": {"padding": "5!important", "padding-top":"0px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"5px"},
    })

if selected =="Data Plat":
    st.title("Data Plat Pelatihan")
    data = st.file_uploader("Upload Data", type=["csv"])
    if data is not None:
        df = pd.read_csv(data)
        st.write(df)
        pilih = st.selectbox('Pilih Gambar',list(df['imagepath']))
        if pilih is not None:
            indeks_abse = df.index[df['imagepath'] == pilih].tolist()
            file_path = df.iloc[indeks_abse[0]][-1]
            x_min = df.iloc[indeks_abse[0]][1]
            y_min = df.iloc[indeks_abse[0]][3]
            x_max = df.iloc[indeks_abse[0]][2]
            y_max = df.iloc[indeks_abse[0]][4]
            
            img = cv2.imread(file_path)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            st.image(img_pil, use_column_width=True)

if selected =="Pelatihan":   
    st.title("Pelatihan")
    df = pd.read_csv('D:/Kuliah/Semester 7/PAI/Deteksi Plat/data_new.csv')
    with st.container():
        col_1, col_2 = st.columns(2)
        with col_1:
            test_size =  st.slider('Jumlah Data test (%):', min_value=1, max_value=50)/100
            learning_rate = st.selectbox('Masukan Jumlah Learning Rate', [1e-4, 1e-3, 1e-2, 1e-1])
        with col_2:
            batch = st.selectbox('Masukan Jumlah batch', [8, 16, 32, 64, 128])
            st.write('')
            epoch = st.selectbox('Masukan Jumlah Epoch', [1, 5, 10, 25, 50, 100, 250])
    nama_model = st.text_input('Masukan nama penyimpanan model terlatih')
    if st.button('Train'):
        with st.spinner('Sedang Melakukan Preprocessing...'):
            data, output = preprocessing(df)
        with st.spinner('Sedang Melakukan Training...'):
            model, history, loss_train, loss_test, average_iou = training(data, output, test_size, learning_rate, batch, epoch)
        st.info(f'Loss Train: {np.round(loss_train, 5)}')
        st.info(f'Loss Test: {np.round(loss_test, 5)}')
        plot_eval(history)
        st.info(f"IoU :{np.round(average_iou*100, 2)} %")
        path_simpan = 'D:/Kuliah/Semester 7/PAI/Deteksi Plat/model'
        model.save(f'{path_simpan}/{nama_model}.h5')
        st.success(f"Model Terlatih '{nama_model}.h5' berhasil disimpan")
        


if selected =="Deteksi Plat":   
    st.title("Deteksi Plat")
    image_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])
    if image_file is not None:
        st.image(image_file, width=400)
    path_simpan = 'D:/Kuliah/Semester 7/PAI/Deteksi Plat/model'
    folder_model = [file for file in os.listdir(path_simpan) if file.endswith('.h5')]
    selected_model = st.selectbox('Pilih Model Terlatih', folder_model)
    file_path_model = os.path.join(path_simpan, selected_model)
    if st.button('Procees'):
        with st.spinner('Sedang Melakukan Deteksi...'):
            xmin, xmax, ymin, ymax, image_with_bbox = detection(image_file, file_path_model)
            pil_image = Image.open(image_file)
            np_image = np.array(pil_image)
            cropped_image = crop_image(np_image, xmin, xmax, ymin, ymax)
            st.image(image_with_bbox, caption='Image with Bounding Box', use_column_width=True)
            st.image(cropped_image, caption='Cropped Image', use_column_width=True)
            text = read_text_from_image(cropped_image)
            st.success(f'Hasil Pembacaan Teks: {text}')