# -*- coding: utf-8 -*-
import cv2
import numpy as np

def detect_fire(frame):
    # Aplica desfoque gaussiano para reduzir o ruído
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    
    # Converte o frame para o espaço de cores HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define o intervalo de cores para detectar fogo (vermelho a laranja)
    lower_bound = np.array([0, 100, 100])
    upper_bound = np.array([10, 255, 255])
    
    # Cria uma máscara para as cores definidas
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Aplica operações morfológicas para remover ruídos
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=4)
    
    # Encontra contornos na máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Marca as áreas onde o fogo foi detectado
    fire_exists = False
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 2000:  # Ajuste este valor para calibrar a sensibilidade
            fire_exists = True
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    return frame, fire_exists

def main():
    # Acessa a câmera (0 é geralmente o índice da câmera padrão)
    cap = cv2.VideoCapture(2)
    
    if not cap.isOpened():
        print("Erro ao acessar a câmera.")
        return
    
    while True:
        # Captura frame a frame
        ret, frame = cap.read()
        
        if not ret:
            print("Falha ao capturar imagem.")
            break
        
        # Detecta fogo no frame
        frame_with_fire_detection, fire_exists = detect_fire(frame)
        
        # Exibe o frame com a detecção de fogo
        cv2.imshow('Camera', frame_with_fire_detection)
        
        # Verifica se há fogo e emite aviso
        if fire_exists:
            print("Fogo detectado!")
        
        # Pressione 'q' para sair do loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Libera a captura e fecha todas as janelas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
