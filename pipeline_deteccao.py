import cv2
import matplotlib.pyplot as plt
import numpy as np

def load_image(path: str):
    image = cv2.imread(path)

    if image is None:
        raise ValueError(f"Imagem não encontrada no caminho: {path}")
    return image
    # Retorna a imagem carregada. Tipo retornado: numpy.ndarray

def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.cvtColor() converte espaço de cor.
    # COLOR_BGR2GRAY: Converter de BGR → Escala de Cinza

def show_image(image, title="Imagem"):
    plt.imshow(image, cmap='gray')
    plt.title(title)   # Define o título do gráfico.
    plt.axis("off")    # Remove eixos 
    plt.show()         # Renderiza e exibe a imagem. Sem isso, nada aparece.


def plot_histogram(image, title="Histograma"):
    plt.hist(image.ravel(), bins=256, range=[0, 256])
    plt.title(title)
    plt.xlabel("Intensidade")          # Eixo X = valor do pixel
    plt.ylabel("Quantidade de Pixels") # Eixo Y = frequência
    plt.show()

def calculate_metrics(image):
    brightness = image.mean()
    contrast = image.std()
    return brightness, contrast

def apply_threshold(image, threshold=127):
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary


def apply_otsu(image):
    _, binary = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary

def apply_morphology(binary_image):

    cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    # MORPH_CLOSE fecha pequenos buracos e conecta regiões próximas
    return cleaned

def find_contours(binary_image):
    contours, _ = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def draw_contours(image, contours):
    image_copy = image.copy()
    cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2)
    return image_copy


def draw_bounding_boxes(image, contours):
    image_copy = image.copy()

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Desenha retângulo ao redor do objeto detectado
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image_copy


def filter_contours_by_area(contours, min_area=500):
    filtered = [c for c in contours if cv2.contourArea(c) >= min_area]
    return filtered


def draw_bounding_boxes_with_area(image, contours):
    image_copy = image.copy()

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        # Desenha o retângulo
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Escreve a área acima do retângulo
        cv2.putText(
            image_copy,
            f"Area: {int(area)}px",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    return image_copy
