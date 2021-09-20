from anomaly_detection import ED_Feedback
import matplotlib.pyplot as plt
from encoders import *
import os
import numpy as np
from tensorflow import keras
from decoders import CNN_Decoder
from anomaly_detection import AutoEncoder

xy_path = "/Users/david/repos/deep-anomaly-detection/data/energy_robot_l400_s2"

train_x = np.load(os.path.join(xy_path, "x.npy"))

vae = AutoEncoder(encoder=CNN_Encoder, decoder=CNN_Decoder,
                  input_shape=train_x.shape, latent_dim=2)

vae.get_model().summary()

#vae.get_model().fit(train_x, train_x, epochs=20, batch_size=20)
#
#keras.utils.plot_model(vae.get_model(), show_shapes=True)
#
#
#plt.plot(train_x[0, :, 0])
#
#pred_x = vae.get_model().predict(train_x[:20, :, :])
#
#plt.plot(pred_x[0, :, 0])
#
# plt.show()

active_ad = ED_Feedback(vae)
active_ad.get_supervised().summary()

keras.utils.plot_model(active_ad.get_supervised())

active_ad.get_supervised().fit(np.zeros((64, 400, 1)), np.zeros((64, 2)), epochs=2)
