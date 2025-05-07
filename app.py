import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# ----- CLASSES DO CIFAR-10 -----
classes = ['Lesões benígnas', 
		'Sinais ou pintas comuns',
		'Tumor benígno de pele',
		'Melanoma',
		'Lesões vasculares',
		'Carcinoma (tipo comum de câncer de pele)',
		'Lesão pré-cancerígena']

# ----- TRANSFORMAÇÃO -----
# transform
## Objeto para pré-processamento na carga
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])
# ----- MODELO (copie aqui sua classe Net) -----
import torch.nn as nn
## forward

## Classe que retorna a rede convolucional
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.LazyLinear(500)

        self.fc2 = nn.Linear(500, 7)

        self.dropout = nn.Dropout(.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        x = self.pool(F.relu(self.conv3(x)))

        x = torch.flatten(x, 1)

        x = self.dropout(x)

        x = F.relu(self.fc1(x))

        x = self.dropout(x)

        x = self.fc2(x)

        return x






# ----- LOAD MODELOS -----
modelo_v1 = Net()
modelo_v1.load_state_dict(torch.load("modelo_final_v1.pt", map_location=torch.device('cpu')))
modelo_v1.eval()

modelo_v2 = Net()
modelo_v2.load_state_dict(torch.load("modelo_final_v2.pt", map_location=torch.device('cpu')))
modelo_v2.eval()

modelo_v3 = Net()
modelo_v3.load_state_dict(torch.load("modelo_final_v3.pt", map_location=torch.device('cpu')))
modelo_v3.eval()

# ----- INTERFACE -----
st.subheader("Classificação de imagens: Identificação de 7 tipos de melasmas")
st.text('''Atenção!!! Este é um trabalho de cunho exclusivamente acedêmico; visa demonstrar a aplicação de técnicas de deep learning  na área de visão computacional. Aliás, nenhum destes modelos obtiveram desempenho minimamente aceitável!
Portanto, nenhuma conclusão médica deve ser tomada considerando o resultado deste trabalho.
Os modelos v1 e v3 sofrem de alto viés provocado pelo desbalanceamento do dataset, e, o modelo v2 sofre de underfitting.''')
st.text('''1 - Procure imagens no google com os seguintes nomes:
                                                    'Lesões benígnas', 
                                                    'Sinais ou pintas comuns',
                                                    'Tumor benígno de pele',
                                                    'Melanoma',
                                                    'Lesões vasculares',
                                                    'Carcinoma (tipo comum de câncer de pele)',
                                                    'Lesão pré-cancerígena;
        2 - Print somente a área que contenha os melasmas, e faça o upload.''')
uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Imagem carregada", use_column_width=True)

    # Prepara imagem
    img_tensor = transform(img).unsqueeze(0)

    modelos = {
        "Modelo v1": modelo_v1,
        "Modelo v2": modelo_v2,
        "Modelo v3": modelo_v3,
    }

    # Layout em 3 colunas
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]

    for (nome_modelo, modelo), col in zip(modelos.items(), cols):
        with torch.no_grad():
            output = modelo(img_tensor)
            pred = output.argmax(dim=1).item()
            probas = torch.softmax(output, dim=1)[0]

        with col:
            st.markdown(f"**{nome_modelo}**")
            st.markdown(f"Classe prevista: **{classes[pred]}**")
            st.bar_chart(probas.numpy())
