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

