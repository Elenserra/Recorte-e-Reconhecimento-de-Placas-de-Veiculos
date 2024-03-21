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

O sistema para detecção de placas de veículos em vídeo, com base na arquitetura You Only Look Once (YOLO) versão 8, foi treinado a partir de um dataset com imagens capturadas por câmeras estáticas localizadas em pedágios [DATASET](https://github.com/raysonlaroca/rodosol-alpr-dataset.git).


## Reconhecimento dos Caracteres das Placas de Veículos

Foi treinado um modelo CNN para realizar o reconhecimento dos caracteres. O mesmo [DATASET](https://github.com/raysonlaroca/rodosol-alpr-dataset.git) foi empregado,  no entanto, foi conduzido um pré-processamento adicional: as imagens das placas foram recortadas e a identificação da placa foi estabelecida pelo nome do arquivo da imagem. Por exemplo, o número da placa "ABC1234" é representado como "ABC1234.jpg". `dataset-preprocess`

