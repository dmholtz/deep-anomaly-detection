import numpy as np
from tensorflow.keras import layers, models


class Decoder(models.Sequential):
    """Maps from the latent space z to the input space x."""

    def __init__(
            self,
            sequence_length,
            num_features,
            name="simple-decoder",
            latent_dim=2,
            **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)

        self.sequence_length = sequence_length
        self.num_features = num_features
        self.latent_dim = latent_dim


class CNN_Decoder(Decoder):

    def __init__(self, sequence_length, num_features, name="cnn-decoder", latent_dim=2, **kwargs):
        super(CNN_Decoder, self).__init__(name=name, sequence_length=sequence_length,
                                          num_features=num_features, latent_dim=latent_dim, **kwargs)

        initial_channels = 64
        upsampling_dim = self.sequence_length // 24 * initial_channels

        self.add(layers.Dense(upsampling_dim, activation='relu'))
        self.add(layers.Reshape((-1, initial_channels)))
        self.add(layers.UpSampling1D(2))
        self.add(layers.Conv1DTranspose(
            filters=32, kernel_size=5, activation='relu', padding='same'))
        self.add(layers.UpSampling1D(2))
        self.add(layers.Conv1DTranspose(
            filters=16, kernel_size=5, activation='relu', padding='same'))
        self.add(layers.UpSampling1D(2))
        self.add(layers.Conv1DTranspose(
            filters=8, kernel_size=5, activation='relu', padding='same'))
        self.add(layers.UpSampling1D(3))
        self.add(layers.Conv1DTranspose(filters=self.num_features,
                 kernel_size=7, activation='relu', padding='same'))


class WaveNet_Decoder(Decoder):

    def __init__(self, sequence_length, num_features, name="wavenet-decoder", latent_dim=2, **kwargs):
        super(WaveNet_Decoder, self).__init__(name=name, sequence_length=sequence_length,
                                              num_features=num_features, latent_dim=latent_dim, **kwargs)

        initial_channels = 64
        upsampling_dim = self.sequence_length // 24 * initial_channels

        self.add(layers.Dense(upsampling_dim, activation='relu'))
        self.add(layers.Reshape((-1, initial_channels)))
        self.add(layers.Conv1DTranspose(filters=32, kernel_size=5,
                 strides=2, activation='relu', padding='same'))
        self.add(layers.Conv1DTranspose(filters=16, kernel_size=5,
                 strides=2, activation='relu', padding='same'))
        self.add(layers.Conv1DTranspose(filters=8, kernel_size=5,
                 strides=2, activation='relu', padding='same'))
        self.add(layers.Conv1DTranspose(filters=self.num_features,
                 kernel_size=5, strides=3, activation='relu', padding='same'))


class LSTM_Decoder(Decoder):

    def __init__(
            self,
            sequence_length,
            num_features,
            name="lstm-decoder",
            latent_dim=2,
            **kwargs):
        super(LSTM_Decoder, self).__init__(name=name, sequence_length=sequence_length,
                                           num_features=num_features, latent_dim=latent_dim, **kwargs)

        self.add(layers.Dense(64, input_dim=self.latent_dim, activation="relu"))
        self.add(layers.RepeatVector(self.sequence_length))
        self.add(layers.LSTM(64, return_sequences=True))
        self.add(layers.LSTM(32, return_sequences=True))
        self.add(layers.TimeDistributed(layers.Dense(self.num_features)))


class CNN_LSTM_Decoder(Decoder):

    def __init__(
            self,
            sequence_length,
            num_features,
            name="cnn-lstm-decoder",
            latent_dim=2,
            **kwargs):
        super(CNN_LSTM_Decoder, self).__init__(name=name, sequence_length=sequence_length,
                                               num_features=num_features, latent_dim=latent_dim, **kwargs)

        upsampling_dim = self.sequence_length // 12

        self.add(layers.Dense(64, input_dim=self.latent_dim, activation="relu"))
        self.add(layers.RepeatVector(upsampling_dim))
        self.add(layers.LSTM(64, return_sequences=True))
        self.add(layers.TimeDistributed(layers.Dense(32)))
        self.add(layers.Conv1DTranspose(filters=16, kernel_size=5,
                 strides=2, activation='relu', padding='same'))
        self.add(layers.Conv1DTranspose(filters=8, kernel_size=5,
                 strides=2, activation='relu', padding='same'))
        self.add(layers.Conv1DTranspose(filters=self.num_features,
                 kernel_size=5, strides=3, activation='relu', padding='same'))


class CNN_LSTM_SA_Decoder(Decoder):

    def __init__(
            self,
            sequence_length,
            num_features,
            name="cnn-lstm-sa-decoder",
            latent_dim=2,
            **kwargs):
        super(CNN_LSTM_SA_Decoder, self).__init__(name=name, sequence_length=sequence_length,
                                                  num_features=num_features, latent_dim=latent_dim, **kwargs)

        upsampling_dim = self.sequence_length // 6

        self.add(layers.Dense(64, input_dim=self.latent_dim, activation="relu"))
        self.add(layers.RepeatVector(upsampling_dim))
        self.add(layers.LSTM(64, return_sequences=True))
        self.add(layers.TimeDistributed(layers.Dense(32)))
        self.add(layers.Conv1DTranspose(filters=8, kernel_size=5,
                 strides=2, activation='relu', padding='same'))
        self.add(layers.Conv1DTranspose(filters=self.num_features,
                 kernel_size=5, strides=3, activation='relu', padding='same'))
