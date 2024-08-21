
import base64
import sys
import os
# Adiciona o diretório pai ao caminho do Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '/../Recorte-e-Reconhecimento-de-Placa/CNN-keras')))

from Model import get_Model
from parameter import letters

# com csv
import io
import contextlib
import itertools
import csv
import hydra
import torch
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator

# Armazenar o valor original de sys.stdout
sys_stdout_temp = sys.stdout
# Restaurar sys.stdout original após a execução do código
@contextlib.contextmanager
def restore_stdout():
    yield
    sys.stdout = sys_stdout_temp

def decode_label(out):
    out_best = list(np.argmax(out[0, 2:], axis=1))
    out_best = [k for k, g in itertools.groupby(out_best)]
    outstr = ''
    for i in out_best:
        if i < len(letters):
            outstr += letters[i]
    return outstr

def predict_license_plate(img, coordinates):
    x, y, w, h = coordinates
    x, y, w, h = int(x), int(y), int(w), int(h)
    img = img[y:h, x:w]
    
    img_pred = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_pred = cv2.resize(img_pred, (128, 64))
    img_pred = (img_pred / 255.0) * 2.0 - 1.0
    img_pred = img_pred.T
    img_pred = np.expand_dims(img_pred, axis=-1)
    img_pred = np.expand_dims(img_pred, axis=0)

    net_out_value = model.predict(img_pred)
    pred_texts = decode_label(net_out_value)

    if len(pred_texts) == 7 and pred_texts[3].isdigit() and (pred_texts[4].isdigit() or pred_texts[4].isalpha()):
        confidences = np.max(net_out_value, axis=2)[0]
        mean_confidence = np.mean(confidences)
        return pred_texts, mean_confidence, net_out_value
    else:
        return None, None, None

# Função para converter imagem para Base64
def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)  # Codificar a imagem como JPEG
    image_base64 = base64.b64encode(buffer).decode('utf-8')  # Converter para Base64 e decodificar para string
    return image_base64

class DetectionPredictor(BasePredictor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.distancia_inicio_metros = cfg.distancia_inicio_metros
        self.pixels_por_metro = cfg.pixels_por_metro
        self.placa_real_width = cfg.placa_real_width
        self.distancia_focal = cfg.distancia_focal
        self.placas_detectadas = {}
        self.frames_bb_dir = self.save_dir / 'frames_with_bounding_boxes'
        self.frames_bb_dir.mkdir(parents=True, exist_ok=True)
        self.detected_plates = {}  # Dicionário para armazenar as placas detectadas

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
        x1, y1, x2, y2 = bbox
        largura_placa_pixels = abs(x2 - x1)
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
    
        with open(os.path.join(self.save_dir, 'resultados.csv'), 'a', newline='') as csvfile:
            fieldnames = ['Frame', 'Placa', 'Distância', 'Confiança']
            csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
            # Se o arquivo estiver vazio, escreva o cabeçalho
            if csvfile.tell() == 0:
                csv_writer.writeheader()
    
            plate_data = {}  # Dicionário para armazenar as informações das placas detectadas
    
            # Ler as entradas existentes no arquivo CSV e atualizar o dicionário
            with open(os.path.join(self.save_dir, 'resultados.csv'), 'r') as csv_existing:
                existing_reader = csv.reader(csv_existing)
                try:
                    next(existing_reader)  # Pular cabeçalho
                except StopIteration:
                    # Se o arquivo estiver vazio, não há necessidade de fazer nada
                    pass
                else:
                    for row in existing_reader:
                        plate = row[1]  # Placa está na segunda coluna 
                        confidence = float(row[3])  # Confiança está na quarta coluna 
                        
                        if plate in plate_data:
                            if confidence > plate_data[plate]["Confiança"]:
                                plate_data[plate] = {"Distância": row[2], "Confiança": confidence}
                        else:
                            plate_data[plate] = {"Distância": row[2], "Confiança": confidence}
        
            for *xyxy, conf, cls in reversed(det):
                if self.args.save or self.args.save_crop or self.args.show:
                    c = int(cls)
                    label = None if self.args.hide_labels else (
                        self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                    distancia_placa = self.calcular_distancia_placa(xyxy)
    
                    if 0.40 <= distancia_placa <= 0.71:
                        cv2.rectangle(im0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                        text_ocr, confianca_atual, net_out_value = predict_license_plate(im0, xyxy)
    
                        if text_ocr is not None:
                            if text_ocr not in plate_data:
                                # Se a placa não está no dicionário, adicione-a
                                if confianca_atual >= 0.96:
                                    plate_data[text_ocr] = {"Distância": f'{distancia_placa:.2f} metros', "Confiança": f'{confianca_atual:.2f}'}
                                    csv_writer.writerow({
                                            'Frame': f'{p.stem}_frame_{frame}',
                                            'Placa': text_ocr,
                                            'Distância': f'{distancia_placa:.2f} metros',
                                            'Confiança': f'{confianca_atual:.2f}'
                                    })
                                    frame_save_path = self.frames_bb_dir / f"{p.stem}_frame_{frame}.jpg"
                                    cv2.imwrite(str(frame_save_path), im0)

                                    # Adicionar placa detectada ao dicionário JSON
                                    detection_time = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(frame / 30))  # Assumindo 30 FPS
                                    photo_path = str(frame_save_path)  # Caminho da foto
                                    latitude = 0.0  ##
                                    longitude = 0.0  ##
                                    
                                    # Converter imagem para Base64
                                    image_base64 = image_to_base64(im0)

                                    self.detected_plates[text_ocr] = {
                                        "timestamp": detection_time,
                                        "latitude": latitude,
                                        "longitude": longitude,
                                        "photo": image_base64
                                    }

                                    # Salvar dicionário no arquivo JSON
                                    with open(os.path.join(self.save_dir, 'detected_plates.json'), 'w') as json_file:
                                        json.dump(self.detected_plates, json_file, indent=4)
                            else:
                                # Se a placa já está no dicionário, atualize as informações se necessário
                                if confianca_atual > float(plate_data[text_ocr]["Confiança"]):
                                    plate_data[text_ocr] = {"Distância": f'{distancia_placa:.2f} metros', "Confiança": f'{confianca_atual:.2f}'}
    
            return log_string


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()

if __name__ == "__main__":
    model = get_Model(training=False)
    model.load_weights("/../Recorte-e-Reconhecimento-de-Placa/CNN-keras/LSTM+BN5---test4.hdf5")
    with restore_stdout():
        predict()
