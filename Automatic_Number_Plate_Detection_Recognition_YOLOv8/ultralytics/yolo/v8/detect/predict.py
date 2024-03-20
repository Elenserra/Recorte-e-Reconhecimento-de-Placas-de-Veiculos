

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

