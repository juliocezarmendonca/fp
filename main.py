from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import base64
import numpy as np
import time

# Inicializar o Flask
app = Flask(__name__)

# Carregar o modelo YOLOv8 uma vez ao inicializar a API
inicio = time.time()
model = YOLO('best2.pt')  # O modelo será carregado uma única vez ao iniciar a API
fim = time.time()
tempo_total = fim - inicio
print(f"Tempo de carregamento do modelo: {tempo_total:.2f} segundos")

# Função para recortar imagens com base nas detecções do YOLOv8
def cut_image(img, model):
    # Realizar a inferência na imagem
    results = model(img)
    cropped_images = []
    confidences = []
    count = 0

    # Iterar sobre os resultados e extrair as detecções
    for result in results:
        for box in result.boxes:
            # Coordenadas do retângulo (x_min, y_min, x_max, y_max)
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            cropped_img = img[y_min:y_max, x_min:x_max]  # Recortar a imagem
            cropped_images.append(cropped_img)

            # Obter a confiança da detecção
            confidence = round(float(box.conf[0]), 2)
            confidences.append(confidence)
            count += 1

    return cropped_images, count, confidences

# Rota para fazer a inferência recebendo uma imagem em base64
@app.route('/infer', methods=['POST'])
def infer_image():
    # Ler a imagem em base64 do JSON recebido
    data = request.json
    image_base64 = data.get('image')
