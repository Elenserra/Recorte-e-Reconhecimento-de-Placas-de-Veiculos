#  Recorte e Reconhecimento de Placa

## Etapas para executar o código

- Clonar o repositório
     
      git clone https://github.com/Elenserra/Recorte-e-Reconhecimento-de-Placa.git
  
- Vá para a pasta clonada
  
      cd Automatic_Number_Plate_Detection_Recognition_YOLOv8

- Instale as dependências

      pip install -e '.[dev]'

- Para detecção e reconhecimento de placas de veículos

      python predict.py model='/../Automatic_Number_Plate_Detection_Recognition_YOLOv8/runs/detect/train/weights/best.pt' source='caminho-do-video.mp4'


## Detecção de Placas de Veículos com YOLOv8n

O sistema para detecção de placas de veículos em vídeo, com base na arquitetura You Only Look Once (YOLO) versão 8, foi treinado a partir de um dataset com imagens capturadas por câmeras estáticas localizadas em pedágios [dataset](https://github.com/raysonlaroca/rodosol-alpr-dataset.git).

