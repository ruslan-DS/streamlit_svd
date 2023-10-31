import pandas as pd
import numpy as np
from matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.decomposition import TruncatedSVD
from skimage import io


# st.set_page_config(layout='wide')

st.write("""
 # Конструктор SVD-метода
""")

st.write("""
 #### Попробуй осуществить один из самых популярных линейных методов машинного обучени - SVD-метод \
    с помощью Slider.
 """)

image_from_file = st.file_uploader(label='Пожалуйста, перед тем, как начнешь играться с конструктором, загрузи любую черно-белую картинку')

st.write("""
 - Примечание:
""")
st.info('Интересено, что мы можем представить каждое изображение в виде многомерной матрицы, где каждый её элемент \
         будет равен одному пикселю на данном изображении. \
         Значения каждого элемента будет представлено в формет RGB в диапозоне от 0 до 255.')


if image_from_file:
    image = io.imread(image_from_file)[:, :, 0]
    U, sing_values, V = np.linalg.svd(image)
    sigma = np.zeros(image.shape)
    np.fill_diagonal(sigma, sing_values)

st.write("""
 - Примечание:
""")
st.info('Если ты хочешь увидеть свое изображение в декартовой системе координат:')
button_for_image = st.button('пожалуйста, нажми на кнопку')
if button_for_image:
    plt.imshow(image)
    st.pyplot(plt)

st.write("""
 - Примечание:
""")
st.info('Если ты хочешь увидеть, как будет примерно выглядить матрица пикселей твоего изображения: ')
button_for_matrix = st.button('пожалуйста, нажми на кнопку', key='button_for_matrix')
if button_for_matrix:
    image.shape
    # image[:2, :2]


st.write("""
 ### Работа с констуктором SVD:
""")


top_k = st.slider(label='Выбери свою top_k, чтобы сжать изображение', min_value=0, max_value=max(image.shape), value=image.shape[0])

trunc_U = U[:, :top_k]
trunc_sigma = sigma[:top_k, :top_k]
trunc_V = V[:top_k, :]

cols = st.columns(2)
plt.imshow(image)
cols[0].pyplot(plt)

plt.imshow(trunc_U @ trunc_sigma @ trunc_V)
cols[1].pyplot(plt)

