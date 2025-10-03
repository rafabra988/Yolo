import cv2 as cv
import numpy as np

# Carregar os classificadores em cascata para detecção de rosto e olhos
cascadeFace = cv.CascadeClassifier('myfacedetector.xml')  # Classificador para detecção de rostos
cascadeEye = cv.CascadeClassifier('haarcascade_eye.xml')  # Classificador para detecção de olhos
#cascadeEye = cv.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

# Carregar o vídeo de entrada
captura = cv.VideoCapture('video.mp4')

# Verificar se o vídeo foi aberto corretamente
if not captura.isOpened():
    print("Erro ao abrir o arquivo de vídeo")
    exit()

# Obter as propriedades do vídeo: largura, altura e taxa de quadros
frameLargura = int(captura.get(cv.CAP_PROP_FRAME_WIDTH))
frameAltura = int(captura.get(cv.CAP_PROP_FRAME_HEIGHT))
frameFPS = int(captura.get(cv.CAP_PROP_FPS))

# Configurar o vídeo de saída com as mesmas dimensões e taxa de quadros do vídeo de entrada
fourcc = cv.VideoWriter_fourcc(*'XVID')  # Codec de vídeo para o arquivo de saída
output = cv.VideoWriter('NOVOvideo.avi', fourcc, frameFPS, (frameLargura, frameAltura))

while True:
    # Ler o próximo quadro do vídeo
    ret, frame = captura.read()
    if not ret:
        break  # Encerra o loop se não houver mais quadros disponíveis

    # Converter o quadro para escala de cinza (necessário para detecção)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detectar rostos no quadro
    faces = cascadeFace.detectMultiScale(gray_img)
    # Detectar olhos no quadro
    eyes = cascadeEye.detectMultiScale(gray_img)

    # Desenhar retângulos ao redor dos rostos detectados
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

    # Desenhar retângulos ao redor dos olhos detectados
    for (x, y, w, h) in eyes:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)

    # Escrever o quadro processado no vídeo de saída
    output.write(frame)

    # Redimensionar o quadro para exibição (30% do tamanho original)
    largura = int(frame.shape[1] * 0.3)
    altura = int(frame.shape[0] * 0.3)
    redimensionar = cv.resize(frame, (largura, altura))

    # Exibir o quadro na janela
    cv.imshow("Reconhecimento", redimensionar)
    if cv.waitKey(1) == ord('s'):  # Pressione 's' para sair
        break

# Liberar recursos e encerrar janelas após o processamento
output.release()  # Liberar o arquivo de saída
captura.release()  # Liberar o vídeo de entrada
cv.destroyAllWindows()  # Fechar todas as janelas do OpenCV
