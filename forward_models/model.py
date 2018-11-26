import tensorflow as tf
import tensorflow.contrib.eager as tfe


class Normalizer(tf.keras.Model):
    def __init__(self, loc, scale):
        super(Normalizer, self).__init__()
        self.loc = tfe.Variable(loc, trainable=False)
        self.scale = tfe.Variable(scale, trainable=False)

    def call(self, inputs):
        return (inputs - self.loc) / self.scale

    def invert(self, inputs):
        return (inputs * self.scale) + self.loc


class ForwardModel(tf.keras.Model):
    def __init__(self, output_units):
        super(ForwardModel, self).__init__()
        self.dense_embedding = tf.keras.layers.Dense(
            units=64, activation=tf.nn.relu)
        self.initial_state = tfe.Variable(tf.zeros(shape=[64]), trainable=True)
        self.rnn = tf.keras.layers.GRU(
            units=64, return_sequences=True, return_state=True)
        self.dense_logits = tf.keras.layers.Dense(
            units=output_units, activation=None)
        self.state = None

    def get_initial_state(self, batch_size, training=None):
        initial_state = tf.tile(self.initial_state[None, ...], [batch_size, 1])
        if training:
            initial_state += tf.random_normal(initial_state.shape, stddev=1e-3)
        return [initial_state]

    def call(self, states, actions, training=None, reset_state=None):
        inputs = tf.concat([states, actions], axis=-1)
        hidden = self.dense_embedding(inputs)

        if self.state is None or reset_state:
            self.state = self.get_initial_state(
                inputs.shape[0], training=training)

        hidden, self.state = self.rnn(hidden, initial_state=self.state)
        logits = self.dense_logits(hidden)
        return logits
