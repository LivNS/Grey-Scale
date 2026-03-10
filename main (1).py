from pipeline_deteccao import (
    load_image,
    convert_to_gray,
    show_image,
    plot_histogram,
    calculate_metrics,
    apply_threshold,
    apply_otsu,
    apply_morphology,
    find_contours,
    draw_contours,
    draw_bounding_boxes,
    filter_contours_by_area,
    draw_bounding_boxes_with_area
)

if __name__ == "__main__":
    img = load_image("sample.jpg")
    print("Shape imagem colorida:", img.shape)
    gray = convert_to_gray(img)
    print("Shape imagem grayscale:", gray.shape)

    show_image(gray, "1 - Imagem Original (Escala de Cinza)")
    plot_histogram(gray, "1 - Histograma Original")

    brightness, contrast = calculate_metrics(gray)

    print("\n--- MÉTRICAS ORIGINAIS ---")
    print(f"Brilho médio:             {brightness:.2f}")
    print(f"Contraste (desvio padrão): {contrast:.2f}")

    binary_manual = apply_threshold(gray, threshold=120)
    show_image(binary_manual, "2 - Threshold Manual (120)")
    plot_histogram(binary_manual, "2 - Histograma - Threshold Manual")

    binary_otsu = apply_otsu(gray)
    show_image(binary_otsu, "3 - Threshold Otsu (Automático)")
    plot_histogram(binary_otsu, "3 - Histograma - Otsu")

    cleaned = apply_morphology(binary_otsu)
    show_image(cleaned, "4 - Após Morfologia (Fechamento)")

    contours_raw = find_contours(cleaned)
    print(f"\nObjetos detectados (sem filtro): {len(contours_raw)}")

    img_contours = draw_contours(img, contours_raw)
    show_image(img_contours, "5 - Contornos Detectados")

    img_boxes = draw_bounding_boxes(img, contours_raw)
    show_image(img_boxes, "6 - Bounding Boxes")

    contours_filtered = filter_contours_by_area(contours_raw, min_area=500)
    print(f"Objetos após filtro de área (>= 500px²): {len(contours_filtered)}")

    img_boxes_area = draw_bounding_boxes_with_area(img, contours_filtered)
    show_image(img_boxes_area, "7 - Bounding Boxes com Área (BÔNUS)")

    print("\n" + "="*50)
    print("RESULTADO FINAL DO PIPELINE")
    print("="*50)
    print(f"Total de objetos detectados: {len(contours_filtered)}")
    print("="*50)