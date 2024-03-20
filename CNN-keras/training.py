from tensorflow.keras.layers import concatenate
from keras import backend as K
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from Image_Generator import TextImageGenerator
from Model import get_Model
from parameter import *

K.set_learning_phase(0)

# Model description and training
model = get_Model(training=True)

try:
    model.load_weights('LSTM+BN4--26--0.011.hdf5')
    print("...Previous weight data...")
except:
    print("...New weight data...")
    pass

train_file_path = './DB/train/'
tiger_train = TextImageGenerator(train_file_path, img_w, img_h, batch_size, downsample_factor)
tiger_train.build_data()

valid_file_path = './DB/test/'
tiger_val = TextImageGenerator(valid_file_path, img_w, img_h, val_batch_size, downsample_factor)
tiger_val.build_data()

# Otimizador com uma taxa de aprendizado ajustada
ada = Adadelta(learning_rate=1.0, rho=0.95, epsilon=1e-07)

early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=4, mode='min', verbose=1)

# Define o caminho para salvar os pesos
checkpoint = ModelCheckpoint(filepath='LSTM+BN5--epoch.hdf5', monitor='val_loss', verbose=1, mode='min', save_best_only=True)

# Reduz a taxa de aprendizado se a perda de validação parar de melhorar
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=0.00001, verbose=1)

# Compila o modelo
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=ada, metrics=['accuracy'])

# Treina o modelo
model.fit_generator(generator=tiger_train.next_batch(),
                    steps_per_epoch=int(tiger_train.n / batch_size),
                    epochs=100,
                    callbacks=[checkpoint, early_stop, reduce_lr],
                    validation_data=tiger_val.next_batch(),
                    validation_steps=int(tiger_val.n / val_batch_size))
