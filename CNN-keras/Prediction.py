
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
model.load_weights("/home/elenserra/LSDi/CRNN-Keras/LSTM+BN5--epoch.hdf5")

# Pasta contendo as imagens de teste
folder_path = "/home/elenserra/LSDi/CRNN-Keras/DB/val/"

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

# import cv2
# import numpy as np
# import itertools
# import os
# from Model import get_Model
# from parameter import letters

# Region = {"A": "A", "B": "B", "C": "C", "D": "D", "E": "E", "F": "F",
#           "G": "G", "H": "H", "I": "I", "J": "J", "K": "K", "L": "L",
#           "M": "M", "N": "N", "O": "O", "P": "P", "Q": "Q", "R": "R",
#           "S": "S", "T": "T", "U": "U", "V": "V", "W": "W", "X": "X",
#           "Y": "Y", "Z": "Z",
#           "0": "0", "1": "1", "2": "2", "3": "3", "4": "4", 
#           "5": "5", "6": "6", "7": "7", "8": "8", "9": "9"}

# def decode_label(out):
#     out_best = list(np.argmax(out[0, 2:], axis=1))
#     out_best = [k for k, g in itertools.groupby(out_best)]
#     outstr = ''
#     for i in out_best:
#         if i < len(letters):
#             outstr += letters[i]
#     return outstr

# def label_to_region(label):
#     region = label[0]
#     two_num = label[1:3]
#     four_num = label[3:]
    
#     try:
#         region = Region[region] if region in Region else ''
#     except:
#         pass
#     return region + two_num + four_num

# # Carregar o modelo treinado
# model = get_Model(training=False)
# model.load_weights("/home/elenserra/LSDi/CRNN-Keras/LSTM+BN5--last_epoch.hdf5")

# # Pasta contendo as imagens de teste
# folder_path = "/home/elenserra/LSDi/CRNN-Keras/DB/test/"

# # Listar todos os arquivos na pasta
# image_files = os.listdir(folder_path)

# # Iterar sobre cada imagem na pasta
# for image_file in image_files:
#     # Construir o caminho completo da imagem
#     image_path = os.path.join(folder_path, image_file)
    
#     # Carregar a imagem de teste
#     img = cv2.imread(image_path)
    
#     # Verificar se a imagem foi carregada corretamente
#     if img is None:
#         print("Erro ao carregar a imagem:", image_file)
#         continue

#     # Extrair o valor real da placa do nome do arquivo
#     real_plate = os.path.splitext(image_file)[0]

#     # Pré-processamento da imagem
#     img_pred = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converter para escala de cinza
#     img_pred = cv2.resize(img_pred, (128, 64))
#     img_pred = (img_pred / 255.0) * 2.0 - 1.0
#     img_pred = img_pred.T
#     img_pred = np.expand_dims(img_pred, axis=-1)
#     img_pred = np.expand_dims(img_pred, axis=0)

#     # Fazer a previsão
#     net_out_value = model.predict(img_pred)
#     pred_texts = decode_label(net_out_value)

#     # Mostrar o resultado
#     #print("Imagem:", image_file)
#     print("Placa real:", real_plate)
#     print("Placa prevista:", label_to_region(pred_texts))
#     print()



#-------------------------------------

# import cv2
# import itertools, os, time
# import numpy as np
# from Model import get_Model
# from parameter import letters
# import argparse
# from keras import backend as K
# K.set_learning_phase(0)

# Region = {"A": "Acre", "B": "Alagoas", "C": "Amapá", "D": "Amazonas", "E": "Bahia", "F": "Ceará",
#           "G": "Distrito Federal", "H": "Espírito Santo", "I": "Goiás", "J": "Maranhão", "K": "Mato Grosso",
#           "L": "Mato Grosso do Sul", "M": "Minas Gerais", "N": "Pará", "O": "Paraíba", "P": "Paraná",
#           "Q": "Pernambuco", "R": "Piauí", "S": "Rio de Janeiro", "T": "Rio Grande do Norte",
#           "U": "Rio Grande do Sul", "V": "Rondônia", "W": "Roraima", "X": "Santa Catarina",
#           "Y": "São Paulo", "Z": "Sergipe"}
# Hangul = {}  # Removendo mapeamento Hangul para o alfabeto coreano

# def decode_label(out):
#     # out : (1, 32, 42)
#     out_best = list(np.argmax(out[0, 2:], axis=1))  # get max index -> len = 32
#     out_best = [k for k, g in itertools.groupby(out_best)]  # remove overlap value
#     outstr = ''
#     for i in out_best:
#         if i < len(letters):
#             outstr += letters[i]
#     return outstr


# def label_to_hangul(label):  # eng -> hangul
#     region = label[0]
#     two_num = label[1:3]
#     four_num = label[3:]

#     try:
#         region = Region[region] if region in Region else ''
#     except:
#         pass
#     return region + two_num + four_num

# parser = argparse.ArgumentParser()
# parser.add_argument("-w", "--weight", help="weight file directory",
#                     type=str, default="LSTM+BN5--last_epoch.hdf5")
# parser.add_argument("-t", "--test_img", help="Test image directory",
#                     type=str, default="./DB/test/")
# args = parser.parse_args()

# # Get CRNN model
# model = get_Model(training=False)

# try:
#     model.load_weights(args.weight)
#     print("...Previous weight data...")
# except:
#     raise Exception("No weight file!")

# test_dir = args.test_img
# test_imgs = os.listdir(args.test_img)
# total = 0
# acc = 0
# letter_total = 0
# letter_acc = 0
# start = time.time()

# for test_img in test_imgs:
#     img = cv2.imread(test_dir + test_img, cv2.IMREAD_GRAYSCALE)

#     img_pred = img.astype(np.float32)
#     img_pred = cv2.resize(img_pred, (128, 64))
#     img_pred = (img_pred / 255.0) * 2.0 - 1.0
#     img_pred = img_pred.T
#     img_pred = np.expand_dims(img_pred, axis=-1)
#     img_pred = np.expand_dims(img_pred, axis=0)

#     net_out_value = model.predict(img_pred)

#     pred_texts = decode_label(net_out_value)

#     for i in range(min(len(pred_texts), len(test_img[0:-4]))):
#         if pred_texts[i] == test_img[i]:
#             letter_acc += 1
#     letter_total += max(len(pred_texts), len(test_img[0:-4]))

#     if pred_texts == test_img[0:-4]:
#         acc += 1
#     total += 1
#     print('Previsto: %s  /  Verdadeiro: %s' % (label_to_hangul(pred_texts), label_to_hangul(test_img[0:-4])))

# end = time.time()
# total_time = (end - start)
# print("Tempo: ", total_time / total)
# print("Acurácia: ", acc / total)
# print("Acurácia das letras: ", letter_acc / letter_total)
