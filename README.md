#  Recorte e Reconhecimento de Placas de Veículos

## Etapas para executar o código

- Clonar o repositório
     
      git clone https://github.com/Elenserra/Recorte-e-Reconhecimento-de-Placa.git
  
- Vá para a pasta clonada
  
      cd Automatic_Number_Plate_Detection_Recognition_YOLOv8

- Configurar e instalar as dependências

       import os
       os.environ['HYDRA_FULL_ERROR'] = '1'
  
       pip install -e '.[dev]'

- Para detecção e reconhecimento de placas de veículos (lembre de ajustar o caminho do modelo de reconhecimento dos caracteres `LSTM+BN5--epoch.hdf5` no script `predict.py`)

      python predict.py model='/../Automatic_Number_Plate_Detection_Recognition_YOLOv8/runs/detect/train/weights/best.pt' source='caminho-do-video.mp4'


## Detecção de Placas de Veículos com YOLOv8n

O sistema para detecção de placas de veículos em vídeo, com base na arquitetura You Only Look Once (YOLO) versão 8, foi treinado a partir de um dataset com imagens capturadas por câmeras estáticas localizadas em pedágios [DATASET](https://github.com/raysonlaroca/rodosol-alpr-dataset.git).


## Reconhecimento dos Caracteres das Placas de Veículos

Foi treinado um modelo CNN para realizar o reconhecimento dos caracteres. O mesmo [DATASET](https://github.com/raysonlaroca/rodosol-alpr-dataset.git) foi empregado,  no entanto, foi conduzido um pré-processamento adicional: as imagens das placas foram recortadas e a descrição da placa passou a ser o nome do arquivo da imagem. Por exemplo, a descrição da placa "ABC1234" é representado como "ABC1234.jpg" no script `dataset_preprocess.py`. E os dados foram separados em 80/10/10, treinamento, validação e teste, respectivamente.

Além disso, foi utilizado a rede neural [CRNN](https://github.com/qjadud1994/CRNN-Keras.git)(combinação entre CNN e RNN), para o reconhecimento dos caracteres das placas.

