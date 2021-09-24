import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.python.keras import activations


class Encoder():
    """Maps from the latent space z to the input space x."""

    def __init__(
            self,
            input,  # keras.Input
            name="simple-encoder",
            latent_dim=2,
            is_variational=True):

        self.name = name
        self.input = input
        self.latent_dim = latent_dim
        self.is_variational = is_variational

        self.build_model()

    def _encoder_architecture(self, input):
        """Declare the encoder's architecture using the functional API"""
        return input

    def build_model(self):

        x = self.input
        h = self._encoder_architecture(x)

        if self.is_variational:
            #h = layers.BatchNormalization()(h)
            z_mean = layers.Dense(self.latent_dim)(h)
            z_log_var = layers.Dense(self.latent_dim)(h)
            z_mean, z_log_var = KLDivergenceLayer(
                weight=0.0002)([z_mean, z_log_var])
            self.z = SamplingLayer(epsilon_std=1)(
                [z_mean, z_log_var])
        else:
            self.z = layers.Dense(self.latent_dim, activation='relu')(h)

        self.model = keras.Model(
            inputs=self.input, outputs=self.z, name=self.name)

    def attention_model(self):
        raise NotImplementedError("No attention mechanism implemented.")


class CNN_Encoder(Encoder):
    """Maps from the latent space z to the input space x."""

    def __init__(
            self,
            input,  # keras.Input
            name="cnn-encoder",
            latent_dim=2,
            is_variational=True,
            **kwargs):
        super(CNN_Encoder, self).__init__(input, name=name,
                                          latent_dim=latent_dim, is_variational=is_variational, **kwargs)

    def _encoder_architecture(self, input):
        x = layers.Conv1D(filters=8, kernel_size=7, activation="relu")(input)
        x = layers.MaxPool1D(pool_size=3)(x)
        x = layers.Conv1D(filters=16, kernel_size=5, activation="relu")(x)
        x = layers.MaxPool1D(pool_size=2)(x)
        x = layers.Conv1D(filters=32, kernel_size=5, activation="relu")(x)
        x = layers.MaxPool1D(pool_size=2)(x)
        x = layers.Conv1D(filters=64, kernel_size=5, activation="relu")(x)
        x = layers.MaxPool1D(pool_size=2)(x)

        return layers.Flatten()(x)


class WaveNet_Encoder(Encoder):
    """Maps from the latent space z to the input space x."""

    def __init__(
            self,
            input,  # keras.Input
            name="wavenet-encoder",
            latent_dim=2,
            is_variational=True,
            **kwargs):
        super(WaveNet_Encoder, self).__init__(input, name=name,
                                              latent_dim=latent_dim, is_variational=is_variational, **kwargs)

    def _encoder_architecture(self, input):

        x = layers.Conv1D(filters=16, kernel_size=5, strides=1,
                          dilation_rate=1, activation="relu", padding="causal")(input)
        x = layers.Conv1D(filters=16, kernel_size=5, strides=1,
                          dilation_rate=2, activation="relu", padding="causal")(x)
        x = layers.Conv1D(filters=16, kernel_size=5, strides=1,
                          dilation_rate=4, activation="relu", padding="causal")(x)
        x = layers.Conv1D(filters=16, kernel_size=5, strides=1,
                          dilation_rate=8, activation="relu", padding="causal")(x)
        x = layers.Conv1D(filters=16, kernel_size=5, strides=1,
                          dilation_rate=16, activation="relu", padding="causal")(x)
        x = layers.Conv1D(filters=16, kernel_size=5, strides=1,
                          dilation_rate=32, activation="relu", padding="causal")(x)
        x = layers.MaxPool1D(2)(x)
        return layers.Flatten()(x)


class LSTM_Encoder(Encoder):
    """Maps from the latent space z to the input space x."""

    def __init__(
            self,
            input,  # keras.Input
            name="lstm-encoder",
            latent_dim=2,
            is_variational=True,
            **kwargs):
        super(LSTM_Encoder, self).__init__(input, name=name,
                                           latent_dim=latent_dim, is_variational=is_variational, **kwargs)

    def _encoder_architecture(self, input):

        x = layers.LSTM(units=32, return_sequences=True)(input)
        x = layers.LSTM(units=64, return_sequences=False)(x)
        return x


class CNN_LSTM_Encoder(Encoder):
    """Maps from the latent space z to the input space x."""

    def __init__(
            self,
            input,  # keras.Input
            name="cnn-lstm-encoder",
            latent_dim=2,
            is_variational=True,
            **kwargs):
        super(CNN_LSTM_Encoder, self).__init__(input, name=name,
                                               latent_dim=latent_dim, is_variational=is_variational, **kwargs)

    def _encoder_architecture(self, input):

        x = layers.Conv1D(filters=8, kernel_size=7, strides=3,
                          activation='relu', padding='same')(input)
        x = layers.Conv1D(filters=16, kernel_size=5, strides=2,
                          activation='relu', padding='same')(x)
        x = layers.Conv1D(filters=32, kernel_size=5, strides=2,
                          activation='relu', padding='same')(x)
        x = layers.LSTM(units=64, return_sequences=False)(x)
        return x


class CNN_LSTM_SA_Encoder(Encoder):
    """Maps from the latent space z to the input space x."""

    def __init__(
            self,
            input,  # keras.Input
            name="cnn-lstm-encoder",
            latent_dim=2,
            is_variational=True,
            **kwargs):
        super(CNN_LSTM_SA_Encoder, self).__init__(input, name=name,
                                                  latent_dim=latent_dim, is_variational=is_variational, **kwargs)

    def _encoder_architecture(self, input):

        x = layers.Conv1D(filters=8, kernel_size=7, strides=3,
                          activation='relu', padding='same')(input)
        x = layers.Conv1D(filters=16, kernel_size=5, strides=2,
                          activation='relu', padding='same')(x)
        c, self.attention_scores = SelfAttention()(x)

        x = layers.LSTM(units=64, return_sequences=False)(c)
        return x

    def attention_model(self):

        am = models.Model(inputs=self.input, outputs=self.attention_scores)
        return am


class SelfAttention(keras.layers.Layer):

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(SelfAttention, self).__init__(*args, **kwargs)

    def call(self, inputs):

        # scores
        print(f"inputs.shape={inputs.shape}")
        scores = tf.matmul(inputs, inputs, transpose_b=True)
        print(f"scores.shape={scores.shape}")

        # Scale
        dk = inputs.shape[-2]
        scores /= dk

        # Softmax
        scores = tf.nn.softmax(scores, axis=-1)

        # context vector
        context = tf.matmul(scores, inputs)
        print(f"context.shape={context.shape}")

        return inputs, scores


class KLDivergenceLayer(layers.Layer):

    """ Identity transform layer that adds KL divergence to the model loss.
    """

    def __init__(self, kl_name="kl-div", weight=1, *args, **kwargs):
        self.weight = weight  # tf.constant(weight)
        self.kl_name = kl_name
        #self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = -0.5 * tf.reduce_sum(1 + log_var -
                                        tf.square(mu) -
                                        tf.exp(log_var), axis=-1)

        kl_div_loss = self.weight * tf.reduce_mean(kl_batch)

        self.add_loss(kl_div_loss, inputs=inputs)
        self.add_metric(kl_div_loss, name=self.kl_name)

        return inputs

    def get_config(self):
        config = {
            'kl_name': self.kl_name,
            'weight': self.weight,
        }
        return config


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
