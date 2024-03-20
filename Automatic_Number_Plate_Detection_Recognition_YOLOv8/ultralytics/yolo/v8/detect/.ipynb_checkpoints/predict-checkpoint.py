

import sys
import io
import contextlib
import itertools
from pathlib import Path

import hydra
import torch

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

import cv2

import numpy as np
import os

from Model import get_Model
from parameter import letters

# Armazenar o valor original de sys.stdout
sys_stdout_temp = sys.stdout
# Restaurar sys.stdout original após a execução do código
@contextlib.contextmanager
def restore_stdout():
    yield
    sys.stdout = sys_stdout_temp

def decode_label(out):
    out_best = list(np.argmax(out[0, 2:], axis=1))  # Corrigido para atribuir a variável out_best
    out_best = [k for k, g in itertools.groupby(out_best)]
    outstr = ''
    for i in out_best:
        if i < len(letters):
            outstr += letters[i]
    return outstr

def predict_license_plate(img, coordinates):
    x, y, w, h = coordinates
    x, y, w, h = int(x), int(y), int(w), int(h)  # Converter para inteiros
    img = img[y:h, x:w]

    img_pred = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    img_pred = cv2.resize(img_pred, (128, 64))
    img_pred = (img_pred / 255.0) * 2.0 - 1.0
    img_pred = img_pred.T
    img_pred = np.expand_dims(img_pred, axis=-1)
    img_pred = np.expand_dims(img_pred, axis=0)
    
    net_out_value = model.predict(img_pred)
    pred_texts = decode_label(net_out_value)

    # Verificar se a placa atende ao formato desejado (LLLNNNN ou LLLNLNN)
    if len(pred_texts) == 7 and pred_texts[3].isdigit() and (pred_texts[4].isdigit() or pred_texts[4].isalpha()):
        # Obtém a confiança associada a cada letra
        confidences = np.max(net_out_value, axis=2)[0]

        # Calcula a média das confianças de todas as letras
        mean_confidence = np.mean(confidences)

        return pred_texts, mean_confidence, net_out_value

    else:
        # Se a placa não atender ao formato desejado, retorna None
        return None, None, None

class DetectionPredictor(BasePredictor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.distancia_inicio_metros = cfg.distancia_inicio_metros
        self.pixels_por_metro = cfg.pixels_por_metro
        self.placa_real_width = cfg.placa_real_width  # Largura real da placa em metros
        self.distancia_focal = cfg.distancia_focal  # Distância focal da câmera em pixels
        self.placas_detectadas = {}  # Dicionário para armazenar as placas detectadas e seus resultados

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float() 
        img /= 255
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def calcular_distancia_placa(self, bbox):
        # bbox é uma tupla (x1, y1, x2, y2) representando as coordenadas da placa
        x1, y1, x2, y2 = bbox
        largura_placa_pixels = abs(x2 - x1)
        
        # Calcula a distância real da placa usando a fórmula da distância focal
        distancia_real = (self.placa_real_width * self.distancia_focal) / largura_placa_pixels
        
        return distancia_real

    
    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
    
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  
        self.annotator = self.get_annotator(im0)
    
        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string
    
        # Open the .txt file for writing (or create if not exist)
        txt_file_path = os.path.join(self.save_dir, 'resultados.txt')
        os.makedirs(os.path.dirname(txt_file_path), exist_ok=True)  # Ensure the directory exists
        # write
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    
        # Iterate over detections
        for *xyxy, conf, cls in reversed(det):
            if self.args.save_txt:  
                xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() 
        
            if self.args.save or self.args.save_crop or self.args.show:  
                c = int(cls)  
                label = None if self.args.hide_labels else (
                    self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                distancia_placa = self.calcular_distancia_placa(xyxy)  # Calcula a distância para cada detecção
                
                # Verificar se a distância da placa está entre 0.27 e 0.33 metros
                if 0.27 <= distancia_placa <= 0.33:
                    # Fazer a predição e salva
                    text_ocr, confianca_atual, net_out_value = predict_license_plate(im0, xyxy)
                    if text_ocr is not None:
                        # Adicionar ou atualizar a entrada no dicionário de placas detectadas
                        if label not in self.placas_detectadas:
                            self.placas_detectadas[label] = (text_ocr, distancia_placa, confianca_atual)
                        else:
                            # Verificar se a confiança atual é maior do que a anteriormente armazenada
                            if confianca_atual > self.placas_detectadas[label][2]:
                                self.placas_detectadas[label] = (text_ocr, distancia_placa, confianca_atual)
                            
                            # Escrever o resultado em um arquivo
                            with open(txt_file_path, 'w') as f:
                                for placa, info in self.placas_detectadas.items():
                                    plate_text, distance, confidence = info
                                    label_txt = f"Placa: {plate_text}, Distância: {distance:.2f} metros, Confiança: {confidence:.2f}\n"
                                    f.write(label_txt)
                                    print(label_txt)
                                    
                            # Processar o arquivo "resultados.txt" para consolidar informações 
                            plate_data = {}
                            with open(txt_file_path, "r") as file:
                                for line in file:
                                    plate_info = line.strip().split(", ")
                                    plate = plate_info[0].split(": ")[1]
                                    confidence = float(plate_info[2].split(": ")[1])
                                    
                                    if plate in plate_data:
                                        if confidence > plate_data[plate]["Confiança"]:
                                            plate_data[plate] = {"Distância": plate_info[1], "Confiança": confidence}
                                    else:
                                        plate_data[plate] = {"Distância": plate_info[1], "Confiança": confidence}
                        
                            # Escrever as informações consolidadas no arquivo "resultados.txt"
                            with open(txt_file_path, 'w') as f:
                                for plate, info in plate_data.items():
                                    label_txt = f"Placa: {plate}, {info['Distância']}, Confiança: {info['Confiança']}\n"
                                    f.write(label_txt)
                                    print(label_txt)

                            if self.args.save_crop:
                                imc = im0.copy()
                                save_one_box(xyxy,
                                             imc,
                                             file=Path(self.save_dir / 'crops' / str(self.model.model.names[c]) / f'{self.data_path.stem}.jpg'),
                                             BGR=True)
                            
        return log_string
    


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()


if __name__ == "__main__":
    # Carregar o modelo LSTM+BN
    model = get_Model(training=False)
    model.load_weights("/home/elenserra/LSDi/Automatic_Number_Plate_Detection_Recognition_YOLOv8/LSTM+BN5--epoch.hdf5")

    # Restaurar sys.stdout original após a execução do código
    with restore_stdout():
        predict()


#mas ou menos
# import sys
# import io
# import contextlib
# import itertools

# import hydra
# import torch

# from ultralytics.yolo.engine.predictor import BasePredictor
# from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
# from ultralytics.yolo.utils.checks import check_imgsz
# from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

# import cv2

# import numpy as np
# import os

# from Model import get_Model
# from parameter import letters

# # Armazenar o valor original de sys.stdout
# sys_stdout_temp = sys.stdout
# # Restaurar sys.stdout original após a execução do código
# @contextlib.contextmanager
# def restore_stdout():
#     yield
#     sys.stdout = sys_stdout_temp

# def decode_label(out):
#     out_best = list(np.argmax(out[0, 2:], axis=1))  # Corrigido para atribuir a variável out_best
#     out_best = [k for k, g in itertools.groupby(out_best)]
#     outstr = ''
#     for i in out_best:
#         if i < len(letters):
#             outstr += letters[i]
#     return outstr

# def predict_license_plate(img, coordinates):
#     x, y, w, h = coordinates
#     x, y, w, h = int(x), int(y), int(w), int(h)  # Converter para inteiros
#     img = img[y:h, x:w]

#     img_pred = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
#     img_pred = cv2.resize(img_pred, (128, 64))
#     img_pred = (img_pred / 255.0) * 2.0 - 1.0
#     img_pred = img_pred.T
#     img_pred = np.expand_dims(img_pred, axis=-1)
#     img_pred = np.expand_dims(img_pred, axis=0)
    
#     net_out_value = model.predict(img_pred)
#     pred_texts = decode_label(net_out_value)

#     # Verificar se a placa atende ao formato desejado (LLLNNNN ou LLLNLNN)
#     if len(pred_texts) == 7 and pred_texts[3].isdigit() and (pred_texts[4].isdigit() or pred_texts[4].isalpha()):
#         # Obtém a confiança associada a cada letra
#         confidences = np.max(net_out_value, axis=2)[0]

#         # Calcula a média das confianças de todas as letras
#         mean_confidence = np.mean(confidences)

#         return pred_texts, mean_confidence, net_out_value

#     else:
#         # Se a placa não atender ao formato desejado, retorna None
#         return None, None, None

# class DetectionPredictor(BasePredictor):
#     def __init__(self, cfg):
#         super().__init__(cfg)
#         self.distancia_inicio_metros = cfg.distancia_inicio_metros
#         self.pixels_por_metro = cfg.pixels_por_metro
#         self.placa_real_width = cfg.placa_real_width  # Largura real da placa em metros
#         self.distancia_focal = cfg.distancia_focal  # Distância focal da câmera em pixels
#         self.placas_detectadas = {}  # Dicionário para armazenar as placas detectadas e seus resultados

#     def get_annotator(self, img):
#         return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

#     def preprocess(self, img):
#         img = torch.from_numpy(img).to(self.model.device)
#         img = img.half() if self.model.fp16 else img.float() 
#         img /= 255
#         return img

#     def postprocess(self, preds, img, orig_img):
#         preds = ops.non_max_suppression(preds,
#                                         self.args.conf,
#                                         self.args.iou,
#                                         agnostic=self.args.agnostic_nms,
#                                         max_det=self.args.max_det)

#         for i, pred in enumerate(preds):
#             shape = orig_img[i].shape if self.webcam else orig_img.shape
#             pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

#         return preds

#     def calcular_distancia_placa(self, bbox):
#         # bbox é uma tupla (x1, y1, x2, y2) representando as coordenadas da detecção
#         x1, y1, x2, y2 = bbox
#         largura_placa_pixels = abs(x2 - x1)
        
#         # Calcula a distância real da placa usando a fórmula da distância focal
#         distancia_real = (self.placa_real_width * self.distancia_focal) / largura_placa_pixels
        
#         return distancia_real

#     # Dentro da classe DetectionPredictor
    
#     def write_results(self, idx, preds, batch):
#         p, im, im0 = batch
#         log_string = ""
#         if len(im.shape) == 3:
#             im = im[None]  
#         self.seen += 1
#         im0 = im0.copy()
#         if self.webcam:  
#             log_string += f'{idx}: '
#             frame = self.dataset.count
#         else:
#             frame = getattr(self.dataset, 'frame', 0)
    
#         self.data_path = p
#         self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
#         log_string += '%gx%g ' % im.shape[2:]  
#         self.annotator = self.get_annotator(im0)
    
#         det = preds[idx]
#         self.all_outputs.append(det)
#         if len(det) == 0:
#             return log_string
    
#         # Open the .txt file for writing (or create if not exist)
#         txt_file_path = os.path.join(self.save_dir, 'resultados.txt')
#         os.makedirs(os.path.dirname(txt_file_path), exist_ok=True)  # Ensure the directory exists
#         # write
#         gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    
#         # Iterate over detections
#         for *xyxy, conf, cls in reversed(det):
#             if self.args.save_txt:  
#                 xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() 
        
#             if self.args.save or self.args.save_crop or self.args.show:  
#                 c = int(cls)  
#                 label = None if self.args.hide_labels else (
#                     self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
#                 distancia_placa = self.calcular_distancia_placa(xyxy)  # Calcula a distância para cada detecção
                
#                 # Verificar se a distância da placa está entre 0.27 e 0.33 metros
#                 if 0.27 <= distancia_placa <= 0.33:
#                     # Fazer a predição e salvar apenas um resultado para cada placa
#                     text_ocr, confianca_atual, net_out_value = predict_license_plate(im0, xyxy)
#                     if text_ocr is not None:
#                         # Adicionar ou atualizar a entrada no dicionário de placas detectadas
#                         if label not in self.placas_detectadas:
#                             self.placas_detectadas[label] = (text_ocr, distancia_placa, confianca_atual)
                            
#                             # Escrever o resultado em um arquivo
#                             with open(txt_file_path, 'a') as f:
#                                 label_txt = f"Placa: {text_ocr}, Distância: {distancia_placa:.2f} metros, Confiança: {confianca_atual:.2f}\n"
#                                 f.write(label_txt)
#                                 print(label_txt)
                                
#                             if self.args.save_crop:
#                                 imc = im0.copy()
#                                 save_one_box(xyxy,
#                                              imc,
#                                              file=str(self.save_dir / 'crops' / str(self.model.model.names[c]) / f'{self.data_path.stem}.jpg'),
#                                              BGR=True)
        
#         return log_string
    


# @hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
# def predict(cfg):
#     cfg.model = cfg.model or "yolov8n.pt"
#     cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  
#     cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
#     predictor = DetectionPredictor(cfg)
#     predictor()


# if __name__ == "__main__":
#     # Carregar o modelo LSTM+BN
#     model = get_Model(training=False)
#     model.load_weights("/home/elenserra/LSDi/Automatic_Number_Plate_Detection_Recognition_YOLOv8/LSTM+BN5--epoch.hdf5")

#     # Restaurar sys.stdout original após a execução do código
#     with restore_stdout():
#         predict()


#com dstance e repetndo
# import sys
# import io
# import contextlib
# import itertools

# import hydra
# import torch

# from ultralytics.yolo.engine.predictor import BasePredictor
# from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
# from ultralytics.yolo.utils.checks import check_imgsz
# from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

# import cv2

# import numpy as np
# import os

# from Model import get_Model
# from parameter import letters

# # Armazenar o valor original de sys.stdout
# sys_stdout_temp = sys.stdout
# # Restaurar sys.stdout original após a execução do código
# @contextlib.contextmanager
# def restore_stdout():
#     yield
#     sys.stdout = sys_stdout_temp

# def decode_label(out):
#     out_best = list(np.argmax(out[0, 2:], axis=1))  # Corrigido para atribuir a variável out_best
#     out_best = [k for k, g in itertools.groupby(out_best)]
#     outstr = ''
#     for i in out_best:
#         if i < len(letters):
#             outstr += letters[i]
#     return outstr

# def predict_license_plate(img, coordinates):
#     x, y, w, h = coordinates
#     x, y, w, h = int(x), int(y), int(w), int(h)  # Converter para inteiros
#     img = img[y:h, x:w]

#     img_pred = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
#     img_pred = cv2.resize(img_pred, (128, 64))
#     img_pred = (img_pred / 255.0) * 2.0 - 1.0
#     img_pred = img_pred.T
#     img_pred = np.expand_dims(img_pred, axis=-1)
#     img_pred = np.expand_dims(img_pred, axis=0)
    
#     net_out_value = model.predict(img_pred)
#     pred_texts = decode_label(net_out_value)

#     # Verificar se a placa atende ao formato desejado (LLLNNNN ou LLLNLNN)
#     if len(pred_texts) == 7 and pred_texts[3].isdigit() and (pred_texts[4].isdigit() or pred_texts[4].isalpha()):
#         # Obtém a confiança associada a cada letra
#         confidences = np.max(net_out_value, axis=2)[0]

#         # Calcula a média das confianças de todas as letras
#         mean_confidence = np.mean(confidences)

#         return pred_texts, mean_confidence, net_out_value

#     else:
#         # Se a placa não atender ao formato desejado, retorna None
#         return None, None, None

# class DetectionPredictor(BasePredictor):
#     def __init__(self, cfg):
#         super().__init__(cfg)
#         self.distancia_inicio_metros = cfg.distancia_inicio_metros
#         self.pixels_por_metro = cfg.pixels_por_metro
#         self.placa_real_width = cfg.placa_real_width  # Largura real da placa em metros
#         self.distancia_focal = cfg.distancia_focal  # Distância focal da câmera em pixels
#         self.placas_detectadas = set()  # Conjunto para armazenar as placas detectadas

#     def get_annotator(self, img):
#         return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

#     def preprocess(self, img):
#         img = torch.from_numpy(img).to(self.model.device)
#         img = img.half() if self.model.fp16 else img.float() 
#         img /= 255
#         return img

#     def postprocess(self, preds, img, orig_img):
#         preds = ops.non_max_suppression(preds,
#                                         self.args.conf,
#                                         self.args.iou,
#                                         agnostic=self.args.agnostic_nms,
#                                         max_det=self.args.max_det)

#         for i, pred in enumerate(preds):
#             shape = orig_img[i].shape if self.webcam else orig_img.shape
#             pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

#         return preds

#     def calcular_distancia_placa(self, bbox):
#         # bbox é uma tupla (x1, y1, x2, y2) representando as coordenadas da detecção
#         x1, y1, x2, y2 = bbox
#         largura_placa_pixels = abs(x2 - x1)
        
#         # Calcula a distância real da placa usando a fórmula da distância focal
#         distancia_real = (self.placa_real_width * self.distancia_focal) / largura_placa_pixels
        
#         return distancia_real

#     # Dentro da classe DetectionPredictor
    
#     def write_results(self, idx, preds, batch):
#         p, im, im0 = batch
#         log_string = ""
#         if len(im.shape) == 3:
#             im = im[None]  
#         self.seen += 1
#         im0 = im0.copy()
#         if self.webcam:  
#             log_string += f'{idx}: '
#             frame = self.dataset.count
#         else:
#             frame = getattr(self.dataset, 'frame', 0)
    
#         self.data_path = p
#         self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
#         log_string += '%gx%g ' % im.shape[2:]  
#         self.annotator = self.get_annotator(im0)
    
#         det = preds[idx]
#         self.all_outputs.append(det)
#         if len(det) == 0:
#             return log_string
    
#         # Open the .txt file for writing (or create if not exist)
#         txt_file_path = os.path.join(self.save_dir, 'resultados.txt')
#         os.makedirs(os.path.dirname(txt_file_path), exist_ok=True)  # Ensure the directory exists
#         # write
#         gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    
#         # Iterate over detections
#         for *xyxy, conf, cls in reversed(det):
#             if self.args.save_txt:  
#                 xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() 
        
#             if self.args.save or self.args.save_crop or self.args.show:  
#                 c = int(cls)  
#                 label = None if self.args.hide_labels else (
#                     self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
#                 distancia_placa = self.calcular_distancia_placa(xyxy)  # Calcula a distância para cada detecção
                
#                 # Verificar se a distância da placa está entre 0.27 e 0.33 metros
#                 if 0.27 <= distancia_placa <= 0.33:
#                     # Adicionar a placa ao conjunto de placas detectadas
#                     self.placas_detectadas.add(label)
                    
#                     # Fazer a predição e salvar apenas um resultado para cada placa
#                     text_ocr, confianca_atual, net_out_value = predict_license_plate(im0, xyxy)
#                     if text_ocr is not None:
#                         # Escrever o resultado em um arquivo
#                         txt_file_path = os.path.join(self.save_dir, 'resultados.txt')
#                         with open(txt_file_path, 'a') as f:
#                             label_txt = f"Placa: {text_ocr}, Distância: {distancia_placa:.2f} metros, Confiança: {confianca_atual:.2f}\n"
#                             f.write(label_txt)
#                             print(label_txt)
                            
#                         if self.args.save_crop:
#                             imc = im0.copy()
#                             save_one_box(xyxy,
#                                             imc,
#                                             file=str(self.save_dir / 'crops' / str(self.model.model.names[c]) / f'{self.data_path.stem}.jpg'),
#                                             BGR=True)
        
#         return log_string
    


# @hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
# def predict(cfg):
#     cfg.model = cfg.model or "yolov8n.pt"
#     cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  
#     cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
#     predictor = DetectionPredictor(cfg)
#     predictor()


# if __name__ == "__main__":
#     # Carregar o modelo LSTM+BN
#     model = get_Model(training=False)
#     model.load_weights("/home/elenserra/LSDi/Automatic_Number_Plate_Detection_Recognition_YOLOv8/LSTM+BN5--epoch.hdf5")

#     # Restaurar sys.stdout original após a execução do código
#     with restore_stdout():
#         predict()



# com dstance
# import sys
# import io
# import contextlib
# import itertools

# # Definir sys.stdout para usar a codificação UTF-8 e o buffer em linha
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

# import hydra
# import torch

# from ultralytics.yolo.engine.predictor import BasePredictor
# from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
# from ultralytics.yolo.utils.checks import check_imgsz
# from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

# import cv2

# import numpy as np
# import os

# from Model import get_Model
# from parameter import letters

# # Armazenar o valor original de sys.stdout
# sys_stdout_temp = sys.stdout
# # Restaurar sys.stdout original após a execução do código
# @contextlib.contextmanager
# def restore_stdout():
#     yield
#     sys.stdout = sys_stdout_temp

# def decode_label(out):
#     out_best = list(np.argmax(out[0, 2:], axis=1))  # Corrigido para atribuir a variável out_best
#     out_best = [k for k, g in itertools.groupby(out_best)]
#     outstr = ''
#     for i in out_best:
#         if i < len(letters):
#             outstr += letters[i]
#     return outstr

# def predict_license_plate(img, coordinates):
#     x, y, w, h = coordinates
#     x, y, w, h = int(x), int(y), int(w), int(h)  # Converter para inteiros
#     img = img[y:h, x:w]

#     img_pred = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
#     img_pred = cv2.resize(img_pred, (128, 64))
#     img_pred = (img_pred / 255.0) * 2.0 - 1.0
#     img_pred = img_pred.T
#     img_pred = np.expand_dims(img_pred, axis=-1)
#     img_pred = np.expand_dims(img_pred, axis=0)

#     net_out_value = model.predict(img_pred)
#     pred_texts = decode_label(net_out_value)

#     # Verificar se a placa atende ao formato desejado (LLLNNNN ou LLLNLNN)
#     if len(pred_texts) == 7 and pred_texts[3].isdigit() and (pred_texts[4].isdigit() or pred_texts[4].isalpha()):
#         # Obtém a confiança associada a cada letra
#         confidences = np.max(net_out_value, axis=2)[0]

#         # Calcula a média das confianças de todas as letras
#         mean_confidence = np.mean(confidences)

#         return pred_texts, mean_confidence

#     else:
#         # Se a placa não atender ao formato desejado, retorna None
#         return None, None

# class DetectionPredictor(BasePredictor):
#     def __init__(self, cfg):
#         super().__init__(cfg)
#         self.distancia_inicio_metros = cfg.distancia_inicio_metros
#         self.pixels_por_metro = cfg.pixels_por_metro
#         self.placa_real_width = cfg.placa_real_width  # Largura real da placa em metros
#         self.distancia_focal = cfg.distancia_focal  # Distância focal da câmera em pixels

#     def get_annotator(self, img):
#         return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

#     def preprocess(self, img):
#         img = torch.from_numpy(img).to(self.model.device)
#         img = img.half() if self.model.fp16 else img.float() 
#         img /= 255
#         return img

#     def postprocess(self, preds, img, orig_img):
#         preds = ops.non_max_suppression(preds,
#                                         self.args.conf,
#                                         self.args.iou,
#                                         agnostic=self.args.agnostic_nms,
#                                         max_det=self.args.max_det)

#         for i, pred in enumerate(preds):
#             shape = orig_img[i].shape if self.webcam else orig_img.shape
#             pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

#         return preds

#     def calcular_distancia_placa(self, bbox):
#         # bbox é uma tupla (x1, y1, x2, y2) representando as coordenadas da detecção
#         x1, y1, x2, y2 = bbox
#         largura_placa_pixels = abs(x2 - x1)
        
#         # Calcula a distância real da placa usando a fórmula da distância focal
#         distancia_real = (self.placa_real_width * self.distancia_focal) / largura_placa_pixels
        
#         return distancia_real

#     def write_results(self, idx, preds, batch):
#         p, im, im0 = batch
#         log_string = ""
#         if len(im.shape) == 3:
#             im = im[None]  
#         self.seen += 1
#         im0 = im0.copy()
#         if self.webcam:  
#             log_string += f'{idx}: '
#             frame = self.dataset.count
#         else:
#             frame = getattr(self.dataset, 'frame', 0)
    
#         self.data_path = p
#         self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
#         log_string += '%gx%g ' % im.shape[2:]  
#         self.annotator = self.get_annotator(im0)
    
#         det = preds[idx]
#         self.all_outputs.append(det)
#         if len(det) == 0:
#             return log_string
    
#         # Open the .txt file for writing (or create if not exist)
#         txt_file_path = os.path.join(self.save_dir, 'resultados.txt')
#         os.makedirs(os.path.dirname(txt_file_path), exist_ok=True)  # Ensure the directory exists
#         # write
#         gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
#         # Create a dictionary to store results for each license plate
#         results_dict = {}
    
#         # Iterate over detections
#         menor_distancia = float('inf')
#         for *xyxy, conf, cls in reversed(det):
#             if self.args.save_txt:  
#                 xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() 
    
#             if self.args.save or self.args.save_crop or self.args.show:  
#                 c = int(cls)  
#                 label = None if self.args.hide_labels else (
#                     self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
#                 distancia_placa = self.calcular_distancia_placa(xyxy)  # Calcula a distância para cada detecção
#                 if distancia_placa < menor_distancia:
#                     menor_distancia = distancia_placa
#                     text_ocr = predict_license_plate(im0, xyxy)
#                     if text_ocr is not None:
#                         results_dict[text_ocr] = menor_distancia
    
#         # Write the results to the file
#         with open(txt_file_path, 'w') as f:
#             for plate, distance in results_dict.items():
#                 label_txt = f"Placa: {plate}, Distância: {distance:.2f} metros\n"
#                 f.write(label_txt)
#                 print(f"Placa: {plate}, Distância: {distance:.2f} metros\n")
    
#         if self.args.save_crop:
#             imc = im0.copy()
#             save_one_box(xyxy,
#                             imc,
#                             file=str(self.save_dir / 'crops' / str(self.model.model.names[c]) / f'{self.data_path.stem}.jpg'),
#                             BGR=True)
#         return log_string


# @hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
# def predict(cfg):
#     cfg.model = cfg.model or "yolov8n.pt"
#     cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  
#     cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
#     predictor = DetectionPredictor(cfg)
#     predictor()


# if __name__ == "__main__":
#     # Carregar o modelo LSTM+BN
#     model = get_Model(training=False)
#     model.load_weights("/home/elenserra/LSDi/Automatic_Number_Plate_Detection_Recognition_YOLOv8/LSTM+BN5--epoch.hdf5")

#     # Restaurar sys.stdout original após a execução do código
#     with restore_stdout():
#         predict()



# import sys
# import io
# import contextlib
# import itertools

# # Definir sys.stdout para usar a codificação UTF-8 e o buffer em linha
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

# import hydra
# import torch

# from ultralytics.yolo.engine.predictor import BasePredictor
# from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
# from ultralytics.yolo.utils.checks import check_imgsz
# from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

# import cv2

# import numpy as np
# import os

# from Model import get_Model
# from parameter import letters

# # Armazenar o valor original de sys.stdout
# sys_stdout_temp = sys.stdout
# # Restaurar sys.stdout original após a execução do código
# @contextlib.contextmanager
# def restore_stdout():
#     yield
#     sys.stdout = sys_stdout_temp

# def decode_label(out):
#     out_best = list(np.argmax(out[0, 2:], axis=1))  # Corrigido para atribuir a variável out_best
#     out_best = [k for k, g in itertools.groupby(out_best)]
#     outstr = ''
#     for i in out_best:
#         if i < len(letters):
#             outstr += letters[i]
#     return outstr

# def predict_license_plate(img, coordinates):
#     x, y, w, h = coordinates
#     x, y, w, h = int(x), int(y), int(w), int(h)  # Converter para inteiros
#     img = img[y:h, x:w]

#     img_pred = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
#     img_pred = cv2.resize(img_pred, (128, 64))
#     img_pred = (img_pred / 255.0) * 2.0 - 1.0
#     img_pred = img_pred.T
#     img_pred = np.expand_dims(img_pred, axis=-1)
#     img_pred = np.expand_dims(img_pred, axis=0)

#     net_out_value = model.predict(img_pred)
#     pred_texts = decode_label(net_out_value)

#     # Verificar se a placa atende ao formato desejado (LLLNNNN ou LLLNLNN)
#     if len(pred_texts) == 7 and pred_texts[3].isdigit() and (pred_texts[4].isdigit() or pred_texts[4].isalpha()):
#         # Obtém a confiança associada a cada letra
#         confidences = np.max(net_out_value, axis=2)[0]

#         # Calcula a média das confianças de todas as letras
#         mean_confidence = np.mean(confidences)

#         return pred_texts, mean_confidence

#     else:
#         # Se a placa não atender ao formato desejado, retorna None
#         return None, None


# class DetectionPredictor(BasePredictor):
#     def __init__(self, cfg):
#         super().__init__(cfg)
#         self.distancia_inicio_metros = cfg.distancia_inicio_metros
#         self.pixels_por_metro = cfg.pixels_por_metro

#     def get_annotator(self, img):
#         return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

#     def preprocess(self, img):
#         img = torch.from_numpy(img).to(self.model.device)
#         img = img.half() if self.model.fp16 else img.float() 
#         img /= 255
#         return img

#     def postprocess(self, preds, img, orig_img):
#         preds = ops.non_max_suppression(preds,
#                                         self.args.conf,
#                                         self.args.iou,
#                                         agnostic=self.args.agnostic_nms,
#                                         max_det=self.args.max_det)

#         for i, pred in enumerate(preds):
#             shape = orig_img[i].shape if self.webcam else orig_img.shape
#             pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

#         return preds
        
#     def write_results(self, idx, preds, batch):
#         p, im, im0 = batch
#         log_string = ""
#         if len(im.shape) == 3:
#             im = im[None]  
#         self.seen += 1
#         im0 = im0.copy()
#         if self.webcam:  
#             log_string += f'{idx}: '
#             frame = self.dataset.count
#         else:
#             frame = getattr(self.dataset, 'frame', 0)
    
#         self.data_path = p
#         self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
#         log_string += '%gx%g ' % im.shape[2:]  
#         self.annotator = self.get_annotator(im0)
    
#         det = preds[idx]
#         self.all_outputs.append(det)
#         if len(det) == 0:
#             return log_string
    
#         # Open the .txt file for writing (or create if not exist)
#         txt_file_path = os.path.join(self.save_dir, 'resultados.txt')
#         os.makedirs(os.path.dirname(txt_file_path), exist_ok=True)  # Ensure the directory exists
#         # write
#         gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
#         # Create a dictionary to store results for each license plate
#         results_dict = {}
    
#         # Iterate over detections
#         for *xyxy, conf, cls in reversed(det):
#             if self.args.save_txt:  
#                 xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() 
    
#             if self.args.save or self.args.save_crop or self.args.show:  
#                 c = int(cls)  
#                 label = None if self.args.hide_labels else (
#                     self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
#                 text_ocr, mean_confidence = predict_license_plate(im0, xyxy)
#                 if text_ocr is not None and mean_confidence >= 0.95:
#                     label = f"{text_ocr} {mean_confidence:.2f}"  # Adicione a precisão ao rótulo
#                     self.annotator.box_label(xyxy, label, color=colors(c, True))
#                     if text_ocr not in results_dict or mean_confidence > results_dict[text_ocr]:
#                         # Se a placa ainda não estiver no dicionário ou a nova detecção tiver uma confiança maior,
#                         # adicione-a ou atualize-a no dicionário
#                         results_dict[text_ocr] = mean_confidence
    
#         # Read existing results from the file and update if necessary
#         existing_results = {}
#         if os.path.exists(txt_file_path):
#             with open(txt_file_path, 'r') as f:
#                 for line in f:
#                     parts = line.strip().split(', Precisão: ')  # Divide a linha em duas partes
#                     plate = parts[0].split(': ')[1]  # Extrai a placa da primeira parte
#                     confidence = parts[1].split(', Confiança da placa: ')[0]  # Extrai a confiança da segunda parte
#                     existing_results[plate] = float(confidence)

    
#         # Merge existing results with new results and write them to the file
#         with open(txt_file_path, 'w') as f:
#             merged_results = {**existing_results, **results_dict}
#             for plate, confidence in merged_results.items():
#                 label_txt = f"Resultado: {plate}, Precisão: {confidence:.2f}, Confiança da placa: {conf:.2f}\n"
#                 f.write(label_txt)
    
#                 # Print the result for debugging
#                 print(f"Writing to file: {label_txt}")
    
#         if self.args.save_crop:
#             imc = im0.copy()
#             save_one_box(xyxy,
#                             imc,
#                             file=str(self.save_dir / 'crops' / str(self.model.model.names[c]) / f'{self.data_path.stem}.jpg'),
#                             BGR=True)
#         return log_string
# #----------        

#     # def write_results(self, idx, preds, batch):
#     #     p, im, im0 = batch
#     #     log_string = ""
#     #     if len(im.shape) == 3:
#     #         im = im[None]  
#     #     self.seen += 1
#     #     im0 = im0.copy()
#     #     if self.webcam:  
#     #         log_string += f'{idx}: '
#     #         frame = self.dataset.count
#     #     else:
#     #         frame = getattr(self.dataset, 'frame', 0)
    
#     #     self.data_path = p
#     #     self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
#     #     log_string += '%gx%g ' % im.shape[2:]  
#     #     self.annotator = self.get_annotator(im0)
    
#     #     det = preds[idx]
#     #     self.all_outputs.append(det)
#     #     if len(det) == 0:
#     #         return log_string
    
#     #     # Open the .txt file for writing (or create if not exist)
#     #     txt_file_path = os.path.join(self.save_dir, 'resultados.txt')
#     #     os.makedirs(os.path.dirname(txt_file_path), exist_ok=True)  # Ensure the directory exists
#     #     # write
#     #     gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
#     #     with open(txt_file_path, 'a') as f:
#     #         for *xyxy, conf, cls in reversed(det):
#     #             if self.args.save_txt:  
#     #                 xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() 
#     #                 # line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)  
#     #                 # f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    
#     #             if self.args.save or self.args.save_crop or self.args.show:  
#     #                 c = int(cls)  
#     #                 label = None if self.args.hide_labels else (
#     #                     self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
#     #                 text_ocr, mean_confidence = predict_license_plate(im0, xyxy)
#     #                 if text_ocr is not None:
#     #                     label = f"{text_ocr} {mean_confidence:.2f}"  # Adicione a precisão ao rótulo
#     #                     self.annotator.box_label(xyxy, label, color=colors(c, True))
#     #                     if mean_confidence >= 0.95:
#     #                         print(f"Writing to file: {label}")  # Debugging statement
#     #                         # Salvar resultado e confiança em um arquivo .txt
#     #                         label_txt = f"Resultado: {text_ocr}, Precisão: {mean_confidence:.2f}\n"
#     #                         f.write(label_txt)
    
#     #             if self.args.save_crop:
#     #                 imc = im0.copy()
#     #                 save_one_box(xyxy,
#     #                                 imc,
#     #                                 file=str(self.save_dir / 'crops' / str(self.model.model.names[c]) / f'{self.data_path.stem}.jpg'),
#     #                                 BGR=True)
#     #     return log_string


# @hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
# def predict(cfg):
#     cfg.model = cfg.model or "yolov8n.pt"
#     cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  
#     cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
#     predictor = DetectionPredictor(cfg)
#     predictor()

# if __name__ == "__main__":
#     # Carregar o modelo LSTM+BN
#     model = get_Model(training=False)
#     model.load_weights("/home/elenserra/LSDi/Automatic_Number_Plate_Detection_Recognition_YOLOv8/LSTM+BN5--epoch.hdf5")

#     # Restaurar sys.stdout original após a execução do código
#     with restore_stdout():
#         predict()


#gual a zero
# import sys
# import io
# import contextlib
# import itertools

# # Definir sys.stdout para usar a codificação UTF-8 e o buffer em linha
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

# import hydra
# import torch

# from ultralytics.yolo.engine.predictor import BasePredictor
# from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
# from ultralytics.yolo.utils.checks import check_imgsz
# from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

# import cv2

# import numpy as np
# import os

# from Model import get_Model
# from parameter import letters

# # Armazenar o valor original de sys.stdout
# sys_stdout_temp = sys.stdout
# # Restaurar sys.stdout original após a execução do código
# @contextlib.contextmanager
# def restore_stdout():
#     yield
#     sys.stdout = sys_stdout_temp

# def decode_label(out):
#     out_best = list(np.argmax(out[0, 2:], axis=1))  # Corrigido para atribuir a variável out_best
#     out_best = [k for k, g in itertools.groupby(out_best)]
#     outstr = ''
#     for i in out_best:
#         if i < len(letters):
#             outstr += letters[i]
#     return outstr

# def predict_license_plate(img, coordinates):
#     x, y, w, h = int(coordinates[0]), int(coordinates[1]), int(coordinates[2]), int(coordinates[3])
#     img = img[y:h, x:w]

#     img_pred = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
#     img_pred = cv2.resize(img_pred, (128, 64))
#     img_pred = (img_pred / 255.0) * 2.0 - 1.0
#     img_pred = img_pred.T
#     img_pred = np.expand_dims(img_pred, axis=-1)
#     img_pred = np.expand_dims(img_pred, axis=0)

#     net_out_value = model.predict(img_pred)
#     pred_texts = decode_label(net_out_value)

#     # Supondo que a confiança seja a pontuação máxima de saída do modelo
#     confidence = np.max(net_out_value)

#     return pred_texts, confidence


# class DetectionPredictor(BasePredictor):

#     def get_annotator(self, img):
#         return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

#     def preprocess(self, img):
#         img = torch.from_numpy(img).to(self.model.device)
#         img = img.half() if self.model.fp16 else img.float() 
#         img /= 255
#         return img

#     def postprocess(self, preds, img, orig_img):
#         preds = ops.non_max_suppression(preds,
#                                         self.args.conf,
#                                         self.args.iou,
#                                         agnostic=self.args.agnostic_nms,
#                                         max_det=self.args.max_det)

#         for i, pred in enumerate(preds):
#             shape = orig_img[i].shape if self.webcam else orig_img.shape
#             pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

#         return preds

#     def write_results(self, idx, preds, batch):
#         p, im, im0 = batch
#         log_string = ""
#         if len(im.shape) == 3:
#             im = im[None]  
#         self.seen += 1
#         im0 = im0.copy()
#         if self.webcam:  
#             log_string += f'{idx}: '
#             frame = self.dataset.count
#         else:
#             frame = getattr(self.dataset, 'frame', 0)
        
#         self.data_path = p
#         self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
#         log_string += '%gx%g ' % im.shape[2:]  
#         self.annotator = self.get_annotator(im0)
        
#         det = preds[idx]
#         self.all_outputs.append(det)
#         if len(det) == 0:
#             return log_string
    
#         # Abrir o arquivo .txt para gravação (ou criar se não existir)
#         txt_file_path = os.path.join(self.save_dir, 'resultados.txt')
#         with open(txt_file_path, 'a') as f:
#             for *xyxy, conf, cls in reversed(det):
#                 if self.args.save_txt:  
#                     xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() 
#                     line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)  
#                     f.write(('%g ' * len(line)).rstrip() % line + '\n')
        
#                 if self.args.save or self.args.save_crop or self.args.show:  
#                     c = int(cls)  
#                     label = None if self.args.hide_labels else (
#                         self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
#                     text_ocr, confidence = predict_license_plate(im0, xyxy)
#                     if len(text_ocr) == 7:  # Verifica se a predição possui exatamente 7 caracteres
#                         correct_letters = sum(a == b for a, b in zip(text_ocr, self.data_path.stem))
#                         precision = correct_letters / len(text_ocr)
#                         label = f"{text_ocr} {precision:.2f}"  # Adicione a precisão ao rótulo
#                         self.annotator.box_label(xyxy, label, color=colors(c, True))
                        
#                         # Salvar resultado e confiança em um arquivo .txt
#                         label_txt = f"Imagem: {self.data_path.stem}, Resultado: {text_ocr}, Precisão: {precision:.2f}\n"
#                         f.write(label_txt)
        
#                 if self.args.save_crop:
#                     imc = im0.copy()
#                     save_one_box(xyxy,
#                                  imc,
#                                  file=str(self.save_dir / 'crops' / str(self.model.model.names[c]) / f'{self.data_path.stem}.jpg'),
#                                  BGR=True)
        
#         return log_string


# @hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
# def predict(cfg):
#     cfg.model = cfg.model or "yolov8n.pt"
#     cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  
#     cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
#     predictor = DetectionPredictor(cfg)
#     predictor()

# if __name__ == "__main__":
#     # Carregar o modelo LSTM+BN
#     model = get_Model(training=False)
#     model.load_weights("/home/elenserra/LSDi/Automatic_Number_Plate_Detection_Recognition_YOLOv8/LSTM+BN5--epoch.hdf5")

#     # Restaurar sys.stdout original após a execução do código
#     with restore_stdout():
#         predict()




#_____________________________
# import sys
# import io
# import contextlib
# import itertools

# # Definir sys.stdout para usar a codificação UTF-8 e o buffer em linha
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

# import hydra
# import torch

# from ultralytics.yolo.engine.predictor import BasePredictor
# from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
# from ultralytics.yolo.utils.checks import check_imgsz
# from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

# import cv2

# import numpy as np
# import os

# from Model import get_Model
# from parameter import letters

# # Armazenar o valor original de sys.stdout
# sys_stdout_temp = sys.stdout
# # Restaurar sys.stdout original após a execução do código
# @contextlib.contextmanager
# def restore_stdout():
#     yield
#     sys.stdout = sys_stdout_temp

# def decode_label(out):
#     #out_best = list(np.argmax(out[0, 2:], axis=1))
#     out_best = [k for k, g in itertools.groupby(out_best)]
#     outstr = ''
#     for i in out_best:
#         if i < len(letters):
#             outstr += letters[i]
#     return outstr

# def predict_license_plate(img, coordinates):
#     x, y, w, h = int(coordinates[0]), int(coordinates[1]), int(coordinates[2]), int(coordinates[3])
#     img = img[y:h, x:w]

#     img_pred = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
#     img_pred = cv2.resize(img_pred, (128, 64))
#     img_pred = (img_pred / 255.0) * 2.0 - 1.0
#     img_pred = img_pred.T
#     img_pred = np.expand_dims(img_pred, axis=-1)
#     img_pred = np.expand_dims(img_pred, axis=0)

#     net_out_value = model.predict(img_pred)
#     pred_texts = decode_label(net_out_value)

#     return pred_texts

# class DetectionPredictor(BasePredictor):

#     def get_annotator(self, img):
#         return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

#     def preprocess(self, img):
#         img = torch.from_numpy(img).to(self.model.device)
#         img = img.half() if self.model.fp16 else img.float() 
#         img /= 255
#         return img

#     def postprocess(self, preds, img, orig_img):
#         preds = ops.non_max_suppression(preds,
#                                         self.args.conf,
#                                         self.args.iou,
#                                         agnostic=self.args.agnostic_nms,
#                                         max_det=self.args.max_det)

#         for i, pred in enumerate(preds):
#             shape = orig_img[i].shape if self.webcam else orig_img.shape
#             pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

#         return preds

#     def write_results(self, idx, preds, batch):
#         p, im, im0 = batch
#         log_string = ""
#         if len(im.shape) == 3:
#             im = im[None]  
#         self.seen += 1
#         im0 = im0.copy()
#         if self.webcam:  
#             log_string += f'{idx}: '
#             frame = self.dataset.count
#         else:
#             frame = getattr(self.dataset, 'frame', 0)

#         self.data_path = p
#         self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
#         log_string += '%gx%g ' % im.shape[2:]  
#         self.annotator = self.get_annotator(im0)

#         det = preds[idx]
#         self.all_outputs.append(det)
#         if len(det) == 0:
#             return log_string
#         for c in det[:, 5].unique():
#             n = (det[:, 5] == c).sum() 
#             log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        
#         gn = torch.tensor(im0.sha'pe)[[1, 0, 1, 0]]  

#         # Verificar se o diretório de destino existe e, se não, criar
#         if not os.path.exists(os.path.dirname(self.txt_path)):
#             os.makedirs(os.path.dirname(self.txt_path))
        
#         with open(f'{self.txt_path}.txt', 'a') as f:
#             for *xyxy, conf, cls in reversed(det):
#                 if self.args.save_txt:  
#                     xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() 
#                     line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)  
#                     f.write(('%g ' * len(line)).rstrip() % line + '\n')

#                 if self.args.save or self.args.save_crop or self.args.show:  
#                     c = int(cls)  
#                     label = None if self.args.hide_labels else (
#                         self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
#                     text_ocr = predict_license_plate(im0, xyxy)
#                     label = text_ocr              
#                     self.annotator.box_label(xyxy, label, color=colors(c, True))
#                 if self.args.save_crop:
#                     imc = im0.copy()
#                     save_one_box(xyxy,
#                                  imc,
#                                  file=str(self.save_dir / 'crops' / str(self.model.model.names[c]) / f'{self.data_path.stem}.jpg'),
#                                  BGR=True)

#         return log_string

# @hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
# def predict(cfg):
#     cfg.model = cfg.model or "yolov8n.pt"
#     cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  
#     cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
#     predictor = DetectionPredictor(cfg)
#     predictor()

# if __name__ == "__main__":
#     # Carregar o modelo LSTM+BN
#     model = get_Model(training=False)
#     model.load_weights("/home/elenserra/LSDi/Automatic_Number_Plate_Detection_Recognition_YOLOv8/LSTM+BN5--epoch.hdf5")

#     # Restaurar sys.stdout original após a execução do código
#     with restore_stdout():
#         predict()




# import cv2
# import numpy as np
# import torch
# import hydra
# import easyocr
# import sys
# import io
# import os
# import itertools
# from Model import get_Model
# from parameter import letters
# from ultralytics.yolo.engine.predictor import BasePredictor
# from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops, check_imgsz
# from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

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

# def predict_license_plate(img):
#     if img is None:
#         return None

#     img_pred = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
#     img_pred = cv2.resize(img_pred, (128, 64))
#     img_pred = (img_pred / 255.0) * 2.0 - 1.0
#     img_pred = img_pred.T
#     img_pred = np.expand_dims(img_pred, axis=-1)
#     img_pred = np.expand_dims(img_pred, axis=0)

#     net_out_value = model.predict(img_pred)
#     pred_texts = decode_label(net_out_value)

#     return label_to_region(pred_texts)

# def ocr_image(img, coordinates):
#     x, y, w, h = int(coordinates[0]), int(coordinates[1]), int(coordinates[2]), int(coordinates[3])
#     img = img[y:h, x:w]
#     result = predict_license_plate(img)
#     text = ""

#     for res in result:
#         if len(result) == 1:
#             text = res[1]
#         if len(res) > 1 and len(res[1]) > 6 and res[2] > 0.2:
#             text = res[1]
#     #torch.cuda.empty_cache()
#     return str(text)

# class DetectionPredictor(BasePredictor):

#     def get_annotator(self, img):
#         return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

#     def preprocess(self, img):
#         img = torch.from_numpy(img).to(self.model.device)
#         img = img.half() if self.model.fp16 else img.float() 
#         img /= 255
#         return img

#     def postprocess(self, preds, img, orig_img):
#         preds = ops.non_max_suppression(preds,
#                                         self.args.conf,
#                                         self.args.iou,
#                                         agnostic=self.args.agnostic_nms,
#                                         max_det=self.args.max_det)

#         for i, pred in enumerate(preds):
#             shape = orig_img[i].shape if self.webcam else orig_img.shape
#             pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

#         return preds

#     def write_results(self, idx, preds, batch):
#         p, im, im0 = batch
#         log_string = ""
#         if len(im.shape) == 3:
#             im = im[None]  
#         self.seen += 1
#         im0 = im0.copy()
#         if self.webcam:  
#             log_string += f'{idx}: '
#             frame = self.dataset.count
#         else:
#             frame = getattr(self.dataset, 'frame', 0)

#         self.data_path = p
#         self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
#         log_string += '%gx%g ' % im.shape[2:]  
#         self.annotator = self.get_annotator(im0)

#         # Check if directory exists, create if it doesn't
#         os.makedirs(os.path.dirname(self.txt_path), exist_ok=True)

#         det = preds[idx]
#         self.all_outputs.append(det)
#         if len(det) == 0:
#             return log_string
#         for c in det[:, 5].unique():
#             n = (det[:, 5] == c).sum() 
#             log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
#         gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  

#         with open(f'{self.txt_path}.txt', 'a') as f:  
#             for *xyxy, conf, cls in reversed(det):
#                 if self.args.save_txt:  
#                     xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() 
#                     line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)  
#                     f.write(('%g ' * len(line)).rstrip() % line + '\n')

#                 if self.args.save or self.args.save_crop or self.args.show:  
#                     c = int(cls)  
#                     label = None if self.args.hide_labels else (
#                         self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
#                     text_ocr = ocr_image(im0, xyxy)
#                     label = text_ocr              
#                     self.annotator.box_label(xyxy, label, color=colors(c, True))
#                 if self.args.save_crop:
#                     imc = im0.copy()
#                     save_one_box(xyxy,
#                                  imc,
#                                  file=str(self.save_dir / 'crops' / str(self.model.model.names[c]) / f'{self.data_path.stem}.jpg'),
#                                  BGR=True)

#         return log_string

# @hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
# def predict(cfg):
#     cfg.model = cfg.model or "yolov8n.pt"
#     cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  
#     cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
#     predictor = DetectionPredictor(cfg)
#     predictor()

# if __name__ == "__main__":
#     # Load the trained model
#     model = get_Model(training=False)
#     model.load_weights("/home/elenserra/LSDi/Automatic_Number_Plate_Detection_Recognition_YOLOv8/LSTM+BN5--epoch.hdf5")
#     predict()


# import cv2
# import numpy as np
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
# model.load_weights("/home/elenserra/LSDi/Automatic_Number_Plate_Detection_Recognition_YOLOv8/LSTM+BN5--epoch.hdf5")

# def predict_license_plate(img):
#     # Carregar a imagem de teste
#     #img = cv2.imread(image_path)
    
#     # Verificar se a imagem foi carregada corretamente
#     if img is None:
#         #print("Erro ao carregar a imagem:", image_path)
#         return None

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

#     return label_to_region(pred_texts)

# # Ultralytics YOLO 🚀, GPL-3.0 license

# import hydra
# import torch

# from ultralytics.yolo.engine.predictor import BasePredictor
# from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
# from ultralytics.yolo.utils.checks import check_imgsz
# from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

# import easyocr
# import cv2

# import sys
# import io

# # Configurar a codificação para UTF-8
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')  


# def ocr_image(img,coordinates):
#     x,y,w, h = int(coordinates[0]), int(coordinates[1]), int(coordinates[2]),int(coordinates[3])
#     img = img[y:h,x:w]

#     #gray = cv2.cvtColor(img , cv2.COLOR_RGB2GRAY)
#     #gray = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
#     #result = reader.readtext(gray)
#     result = predict_license_plate(img)
#     text = ""

#     for res in result:
#         if len(result) == 1:
#             text = res[1]
#         if len(result) >1 and len(res[1])>6 and res[2]> 0.2:
#             text = res[1]
#     #     text += res[1] + " "
#     torch.cuda.empty_cache()
#     return str(text)

# class DetectionPredictor(BasePredictor):

#     def get_annotator(self, img):
#         return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

#     def preprocess(self, img):
#         img = torch.from_numpy(img).to(self.model.device)
#         img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
#         img /= 255  # 0 - 255 to 0.0 - 1.0
#         return img

#     def postprocess(self, preds, img, orig_img):
#         preds = ops.non_max_suppression(preds,
#                                         self.args.conf,
#                                         self.args.iou,
#                                         agnostic=self.args.agnostic_nms,
#                                         max_det=self.args.max_det)

#         for i, pred in enumerate(preds):
#             shape = orig_img[i].shape if self.webcam else orig_img.shape
#             pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

#         return preds

#     def write_results(self, idx, preds, batch):
#         p, im, im0 = batch
#         log_string = ""
#         if len(im.shape) == 3:
#             im = im[None]  # expand for batch dim
#         self.seen += 1
#         im0 = im0.copy()
#         if self.webcam:  # batch_size >= 1
#             log_string += f'{idx}: '
#             frame = self.dataset.count
#         else:
#             frame = getattr(self.dataset, 'frame', 0)

#         self.data_path = p
#         # save_path = str(self.save_dir / p.name)  # im.jpg
#         self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
#         log_string += '%gx%g ' % im.shape[2:]  # print string
#         self.annotator = self.get_annotator(im0)

#         det = preds[idx]
#         self.all_outputs.append(det)
#         if len(det) == 0:
#             return log_string
#         for c in det[:, 5].unique():
#             n = (det[:, 5] == c).sum()  # detections per class
#             log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
#         # write
#         gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
#         for *xyxy, conf, cls in reversed(det):
#             if self.args.save_txt:  # Write to file
#                 xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
#                 line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)  # label format
#                 with open(f'{self.txt_path}.txt', 'a') as f:
#                     f.write(('%g ' * len(line)).rstrip() % line + '\n')

#             if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
#                 c = int(cls)  # integer class
#                 label = None if self.args.hide_labels else (
#                     self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
#                 text_ocr = ocr_image(im0,xyxy)
#                 label = text_ocr              
#                 self.annotator.box_label(xyxy, label, color=colors(c, True))
#             if self.args.save_crop:
#                 imc = im0.copy()
#                 save_one_box(xyxy,
#                              imc,
#                              file=str(self.save_dir / 'crops' / str(self.model.model.names[c]) / f'{self.data_path.stem}.jpg'),
#                              BGR=True)

#                 # save_one_box(xyxy,
#                 #              imc,
#                 #              file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
#                 #              BGR=True)

#         return log_string


# @hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
# def predict(cfg):
#     cfg.model = cfg.model or "yolov8n.pt"
#     cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
#     cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
#     predictor = DetectionPredictor(cfg)
#     predictor()


# if __name__ == "__main__":
#     predict()
