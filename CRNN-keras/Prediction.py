
import cv2
import numpy as np
import itertools
import os
from Model import get_Model
from parameter import letters
import time

Region = {"A": "A", "B": "B", "C": "C", "D": "D", "E": "E", "F": "F",
          "G": "G", "H": "H", "I": "I", "J": "J", "K": "K", "L": "L",
          "M": "M", "N": "N", "O": "O", "P": "P", "Q": "Q", "R": "R",
          "S": "S", "T": "T", "U": "U", "V": "V", "W": "W", "X": "X",
          "Y": "Y", "Z": "Z",
          "0": "0", "1": "1", "2": "2", "3": "3", "4": "4", 
          "5": "5", "6": "6", "7": "7", "8": "8", "9": "9"}

def decode_label(out):
    out_best = list(np.argmax(out[0, 2:], axis=1))
    out_best = [k for k, g in itertools.groupby(out_best)]
    outstr = ''
    for i in out_best:
        if i < len(letters):
            outstr += letters[i]
    return outstr

def label_to_region(label):
    region = label[0]
    two_num = label[1:3]
    four_num = label[3:]
    
    try:
        region = Region[region] if region in Region else ''
    except:
        pass
    return region + two_num + four_num

# Carregar o modelo treinado
model = get_Model(training=False)
model.load_weights("/../CRNN-Keras/LSTM+BN5--epoch.hdf5")

# Pasta contendo as imagens de teste
folder_path = "/../CRNN-Keras/DB/test/"

# Listar todos os arquivos na pasta
image_files = os.listdir(folder_path)

total_start_time = time.time()

total_correct = 0
total_letters_correct = 0
total_plates = len(image_files)

# Iterar sobre cada imagem na pasta
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    img = cv2.imread(image_path)
    
    if img is None:
        print("Erro ao carregar a imagem:", image_file)
        continue

    real_plate = os.path.splitext(image_file)[0]

    img_pred = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_pred = cv2.resize(img_pred, (128, 64))
    img_pred = (img_pred / 255.0) * 2.0 - 1.0
    img_pred = img_pred.T
    img_pred = np.expand_dims(img_pred, axis=-1)
    img_pred = np.expand_dims(img_pred, axis=0)

    start_time = time.time()

    net_out_value = model.predict(img_pred)
    pred_texts = decode_label(net_out_value)

    end_time = time.time()
    elapsed_time = end_time - start_time

    plate_accuracy = 1 if pred_texts == real_plate else 0
    
    # Calcular a precisão das letras
    if len(pred_texts) == len(real_plate):
        letters_accuracy = sum([1 if pred_texts[i] == real_plate[i] else 0 for i in range(len(real_plate))]) / len(real_plate)
    else:
        letters_accuracy = 0  # Se os comprimentos forem diferentes, definir a precisão das letras como 0
    
    total_correct += plate_accuracy
    total_letters_correct += letters_accuracy

    print("Imagem:", image_file)
    print("Placa real:", real_plate)
    print("Placa prevista:", label_to_region(pred_texts))
    print("Tempo de execução:", elapsed_time, "segundos")
    print("Acurácia da placa:", plate_accuracy)
    print("Acurácia das letras:", letters_accuracy)
    print()

total_end_time = time.time()
total_elapsed_time = total_end_time - total_start_time

total_accuracy = total_correct / total_plates
total_letters_accuracy = total_letters_correct / total_plates

print("Tempo de execução total:", total_elapsed_time, "segundos")
print("Acurácia total das placas:", total_accuracy)
print("Acurácia total das letras:", total_letters_accuracy)

