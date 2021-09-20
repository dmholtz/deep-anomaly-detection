from abc import ABC
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.python.keras import regularizers
from tensorflow.python.ops.gen_array_ops import Reshape, shape


class AutoEncoder():
    """Maps an input x into a latent space z and then computes a reconstruction
    x_hat using z."""

    def __init__(
            self,
            encoder,
            decoder,
            input_shape=(None, 400, 1),
            latent_dim=2,
            is_variational=True):

        # derive dimensions from shape
        assert len(input_shape) == 3
        self.sequence_length = input_shape[1]
        self.num_features = input_shape[2]
        self.latent_dim = latent_dim

        self.input = keras.Input(
            shape=(self.sequence_length, self.num_features))

        # initialize encoder and decoder
        self.encoderWrapper = encoder(
            self.input, latent_dim=latent_dim, is_variational=is_variational)
        self.encoder = self.encoderWrapper.model
        self.decoder = decoder(sequence_length=self.sequence_length,
                               num_features=self.num_features, latent_dim=self.latent_dim)

        # build the AutoEncoder model
        self.is_variational = is_variational
        self._build_ae()

    def _build_ae(self):

        self.z = self.encoder(self.input)
        self.x_hat = self.decoder(self.z)

        self.model = keras.Model(inputs=self.input, outputs=self.x_hat)

        self.model.compile(optimizer=keras.optimizers.Adam(),
                           loss=keras.losses.MeanSquaredError())

    def get_model(self):
        return self.model

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder


class VarEncoder(models.Model):

    def __init__(self, latent_dim):
        super(VarEncoder, self).__init__()
        self.latent_dim = latent_dim

        # self.enc = self.build_encoder()

    def build_encoder(self):
        encoder = models.Sequential(
            [
                layers.InputLayer(input_shape=(
                    480, 4)),
                layers.Conv1D(filters=8, kernel_size=7,
                              strides=3, activation="relu"),
                layers.Conv1D(filters=16, kernel_size=5,
                              strides=2, activation="relu"),
                layers.Conv1D(filters=32, kernel_size=5,
                              strides=2, activation="relu"),
                layers.Conv1D(filters=64, kernel_size=5,
                              strides=2, activation="relu"),
                layers.Flatten(),
                layers.Dense(2*self.latent_dim)
            ]
        )
        return encoder

    def build(self, input_shape):
        self.enc = self.build_encoder()
        self.sampling = SamplingLayer()

    def call(self, inputs):

        h = self.enc(inputs)
        mean, logvar = tf.split(h, num_or_size_splits=2, axis=1)

        # KL-Divergence
        kl_batch = -0.5 * tf.reduce_sum(1 + logvar -
                                        tf.square(mean) -
                                        tf.exp(logvar), axis=-1)
        kl_div_loss = 0.1 * tf.reduce_mean(kl_batch)

        self.add_loss(kl_div_loss)
        self.add_metric(kl_div_loss, name="kl-div")

        z = self.sampling([mean, logvar])
        return z


class AutoEncoder2(keras.Model):
    """Maps an input x into a latent space z and then computes a reconstruction
    x_hat using z."""

    def __init__(
            self,
            encoder,
            decoder,
            input_shape=(None, 400, 1),
            latent_dim=2,
            is_variational=True):

        super(AutoEncoder2, self).__init__()

        # derive dimensions from shape
        assert len(input_shape) == 3
        self.sequence_length = input_shape[1]
        self.num_features = input_shape[2]
        self.latent_dim = latent_dim

        self.is_variational = is_variational

        self.encoder = VarEncoder(latent_dim=latent_dim)
        self.decoder = self.build_decoder()

    def build_encoder(self):
        encoder = models.Sequential(
            [
                layers.InputLayer(input_shape=(
                    self.sequence_length, self.num_features)),
                layers.Conv1D(filters=8, kernel_size=7,
                              strides=3, activation="relu"),
                layers.Conv1D(filters=16, kernel_size=5,
                              strides=2, activation="relu"),
                layers.Conv1D(filters=32, kernel_size=5,
                              strides=2, activation="relu"),
                layers.Conv1D(filters=64, kernel_size=5,
                              strides=2, activation="relu"),
                layers.Flatten(),
                layers.Dense(2*self.latent_dim)
            ]
        )
        return encoder

    def build_decoder(self):
        decoder = models.Sequential(
            [
                layers.InputLayer(input_shape=(self.latent_dim)),
                layers.Dense(20*64, activation='relu'),
                layers.Reshape((-1, 64)),
                layers.Conv1DTranspose(
                    filters=32, kernel_size=5, strides=2, activation="relu", padding='same'),
                layers.Conv1DTranspose(
                    filters=16, kernel_size=5, strides=2, activation="relu", padding='same'),
                layers.Conv1DTranspose(
                    filters=8, kernel_size=5, strides=2, activation="relu", padding='same'),
                layers.Conv1DTranspose(
                    filters=self.num_features, kernel_size=7, strides=3, activation="relu", padding='same'),
            ]
        )
        return decoder

    def call(self, inputs):

        z = self.encoder(inputs)
        # mean, logvar = tf.split(h, num_or_size_splits=2, axis=1)
#
        # KL-Divergence
        # kl_batch = -0.5 * tf.reduce_sum(1 + logvar -
        #                                tf.square(mean) -
        #                                tf.exp(logvar), axis=-1)
        # kl_div_loss = 0.1 * tf.reduce_mean(kl_batch)
#
        # self.add_loss(kl_div_loss)
        # self.add_metric(kl_div_loss, name="kl-div")
#
        # z = SamplingLayer()([mean, logvar])
        x_hat = self.decoder(z)

        loss = keras.losses.MeanSquaredError()(inputs, x_hat)
        # loss = tf.math.square(tf.math.subtract(inputs, x_hat))
        # loss = tf.math.reduce_mean(loss)

        self.add_loss(loss)
        self.add_metric(loss, "mse")
       # x_hat = ReconstructionLossLayer()([inputs, x_hat])

        return x_hat

    def get_model(self):
        return self

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder


class AutoEncoder3(keras.Model):
    """Maps an input x into a latent space z and then computes a reconstruction
    x_hat using z."""

    def __init__(
            self,
            input_shape=(None, 400, 1),
            latent_dim=2,
            is_variational=True):

        super(AutoEncoder3, self).__init__()

        # derive dimensions from shape
        assert len(input_shape) == 3
        self.sequence_length = input_shape[1]
        self.num_features = input_shape[2]
        self.latent_dim = latent_dim

        self.is_variational = is_variational

        self.encoder = self.build_decoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        encoder = models.Sequential(
            [
                layers.InputLayer(input_shape=(
                    self.sequence_length, self.num_features)),
                layers.Conv1D(filters=8, kernel_size=7,
                              strides=3, activation="relu"),
                layers.Conv1D(filters=16, kernel_size=5,
                              strides=2, activation="relu"),
                layers.Conv1D(filters=32, kernel_size=5,
                              strides=2, activation="relu"),
                layers.Conv1D(filters=64, kernel_size=5,
                              strides=2, activation="relu"),
                layers.Flatten(),
                layers.Dense(2*self.latent_dim)
            ]
        )
        return encoder

    def build_decoder(self):
        decoder = models.Sequential(
            [
                layers.InputLayer(input_shape=(self.latent_dim)),
                layers.Dense(20*64, activation='relu'),
                layers.Reshape((-1, 64)),
                layers.Conv1DTranspose(
                    filters=32, kernel_size=5, strides=2, activation="relu", padding='same'),
                layers.Conv1DTranspose(
                    filters=16, kernel_size=5, strides=2, activation="relu", padding='same'),
                layers.Conv1DTranspose(
                    filters=8, kernel_size=5, strides=2, activation="relu", padding='same'),
                layers.Conv1DTranspose(
                    filters=self.num_features, kernel_size=7, strides=3, activation="relu", padding='same'),
            ]
        )
        return decoder

    def call(self, inputs):

        z = self.encoder(inputs)
        mean, logvar = tf.split(h, num_or_size_splits=2, axis=1)

        kl_batch = -0.5 * tf.reduce_sum(1 + logvar -
                                        tf.square(mean) -
                                        tf.exp(logvar), axis=-1)
        kl_div_loss = 0.1 * tf.reduce_mean(kl_batch)

        self.add_loss(kl_div_loss)
        self.add_metric(kl_div_loss, name="kl-div")

        z = SamplingLayer()([mean, logvar])
        x_hat = self.decoder(z)

        loss = keras.losses.MeanSquaredError()(inputs, x_hat)
        self.add_loss(loss)
        self.add_metric(loss, "mse")

        return x_hat

    def get_model(self):
        return self

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder


class ReconstructionLossLayer(layers.Layer):

    def __init__(self, *args, **kwargs):
        super(ReconstructionLossLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        x, x_hat = inputs

        loss = tf.math.square(tf.math.subtract(x, x_hat))
        loss = tf.math.reduce_mean(loss, axis=(1, 2))

        scalar_loss = tf.math.reduce_mean(loss)

        self.add_loss(scalar_loss)
        self.add_metric(scalar_loss, "mse")

        return loss


class BaseActiveAD(ABC):
    """Extends a unsupervised AutoEncoder model by adding a supervised
    model for active anomaly detection."""

    def __init__(self, unsupervised, num_classes=2):

        self.unsupervised = unsupervised
        self.num_classes = num_classes
        self._build_supervised_model()

    def _architecture(self):
        # trivial default implementation
        self.mlp_input = self.unsupervised.input

    def _build_supervised_model(self):

        self.mlp_input = self._architecture()

        self.y = self.mlp_input
        for fc_layer in BaseActiveAD._make_mlp(hidden_layer_dimensions=[16, 8], num_outputs=self.num_classes):
            self.y = fc_layer(self.y)

        self.supervised = keras.Model(inputs=self.input, outputs=self.y)
        self.supervised.compile(
            loss=keras.losses.CategoricalCrossentropy(), optimizer='adam')

    def get_supervised(self):
        return self.supervised

    @ staticmethod
    def _make_mlp(hidden_layer_dimensions=[], num_outputs=2):
        """Creates a mlp given the hidden dimensions and the number of output units."""

        layer_stack = list()
        for dim in hidden_layer_dimensions:
            layer_stack.append(layers.Dense(
                dim, activation='relu', kernel_regularizer=regularizers.l2()))
        layer_stack.append(layers.Dense(num_outputs, activation='softmax'))
        return layer_stack


# class SimpleActiveAD(BaseActiveAD):
#
#    def __init__(self, unsupervised, query):
#        super().__init__(unsupervised, query)
#
#    def _architecture(self):
#        raise NotImplementedError()


class ED_Feedback(BaseActiveAD):

    def __init__(self, unsupervised, num_classes=2):
        super().__init__(unsupervised, num_classes=num_classes)

    def _architecture(self):
        self.input = self.unsupervised.input

        score = ReconstructionLossLayer()(
            [self.input, self.unsupervised.x_hat])
        score = layers.Lambda(lambda x: 0 * x)(score)
        score = layers.Reshape((1,))(score)

        z = self.unsupervised.z

        mlp_input = layers.Concatenate()([z, score])
        return mlp_input


class SamplingLayer(layers.Layer):
    """Samples a point from a normal distribution (mean, log_var)"""

    def __init__(self, epsilon_std=1, *args, **kwargs):
        super(SamplingLayer, self).__init__(*args, **kwargs)
        self.epsilon_std = epsilon_std

    def call(self, inputs):
        mean, log_var = inputs
        epsilon = tf.random.normal(
            shape=tf.shape(mean)) * self.epsilon_std
        return mean + tf.exp(0.5*log_var) * epsilon

    def get_config(self):
        return {"epsilon_std": self.epsilon_std}
