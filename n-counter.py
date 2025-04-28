#Importar librerías
import cv2
import os
from tqdm import tqdm
import csv
from ultralytics import YOLO

def crop_save(model, input_folder, output_folder, confidence_threshold=0.5, max_nuclei=1000):
    
    # Recorta y guarda núcleos hasta alcanzar max_nuclei, con metadatos y barra de progreso
    
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'tiff', 'tif'))]

    # Registro de metadatos
    with open(os.path.join(output_folder, "metadata.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "source_image", "x1", "y1", "x2", "y2", "confidence"])

    # Contadores
    global_counter = 1
    nuclei_count = 0

    # Barra de progreso ajustada a max_nuclei (ojo, no al total de imágenes)
    with tqdm(total=max_nuclei, desc="Recortando núcleos") as pbar:
        for image_file in image_files:
            if nuclei_count >= max_nuclei:
                break

            image_path = os.path.join(input_folder, image_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error al cargar: {image_file}")
                continue

            results = model(image)
            any_detection = False

            for det in results[0].boxes:
                if nuclei_count >= max_nuclei:
                    break

                if det.conf >= confidence_threshold and int(det.cls) == 0:
                    x1, y1, x2, y2 = map(int, det.xyxy[0].cpu().numpy())
                    if (x2 - x1) < 10 or (y2 - y1) < 10:
                        continue

                    nucleus = image[y1:y2, x1:x2]
                    output_filename = f"{global_counter}.png"
                    output_path = os.path.join(output_folder, output_filename)
                    cv2.imwrite(output_path, nucleus)

                    # Guardar metadatos incluyendo la imagen origen
                    with open(os.path.join(output_folder, "metadata.csv"), "a") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            output_filename,
                            image_file,  # Nombre de la imagen original
                            x1, y1, x2, y2,
                            float(det.conf)
                        ])

                    global_counter += 1
                    nuclei_count += 1
                    pbar.update(1)  # Progreso de barra por cada núcleo
                    any_detection = True

            if not any_detection:
                with open(os.path.join(output_folder, "sin_detecciones.txt"), "a") as f:
                    f.write(f"{image_file}\n")

    print(f"\nNúcleos recortados: {nuclei_count}/{max_nuclei}")

# Configuración
INPUT_FOLDER = "C:\Users\Desktop\CARPETA CON FOTOS DE ENTRADA"
OUTPUT_FOLDER = "C:\Users\Desktop\CARPETA CON FOTOS DE SALIDA"
CONFIDENCE_THRESHOLD = 0.7
MAX_NUCLEI = 1000

# Ejecutar
model = YOLO('C:\Users\Desktop\modelo1.pt') # Modelo
crop_save(model, INPUT_FOLDER, OUTPUT_FOLDER, CONFIDENCE_THRESHOLD, MAX_NUCLEI)
