import tensorflow as tf
import tensorflow.contrib.eager as tfe


class Normalizer(tf.keras.Model):
    """
    A simple normalization utility. It stores the args in variables so
    the model and args can be saved/restored with checkpoints.

    Args:
        loc: The mean to use for normalization.
        scale: The standard deviation to use for normalization.
    """

    def __init__(self, loc, scale):
        super(Normalizer, self).__init__()
        self.loc = tfe.Variable(loc, trainable=False)
        self.scale = tfe.Variable(scale, trainable=False)

    def call(self, inputs):
        """
        Normalize the inputs.
        """
        return (inputs - self.loc) / self.scale

    def invert(self, inputs):
        """
        De-normalize the inputs.
        """
        return (inputs * self.scale) + self.loc


class ForwardModel(tf.keras.Model):
    """
    Define the forward model.

    Args:
        output_units: The number of output units for the model.
    """

    def __init__(self, output_units):
        super(ForwardModel, self).__init__()

        # define the initial state of the RNN as a trainable variable so it
        # will be optimized rather than remaining zeros
        self.initial_hidden_state = tfe.Variable(
            tf.zeros(shape=[64]), trainable=True)
        self.initial_cell_state = tfe.Variable(
            tf.zeros(shape=[64]), trainable=True)

        self.dense_embedding = tf.keras.layers.Dense(
            units=64, activation=tf.nn.relu)

        # Note: This does not use a LSTMCell since it would not feed the hidden
        # state from one step to the next. This is important for training so
        # that the model learns how to interpret hidden states rather than
        # ignoring them after each step.

        # using an LSTM but a GRU works about as well and faster
        self.rnn = tf.keras.layers.LSTM(
            units=64, return_sequences=True, return_state=True)
        self.dense_logits = tf.keras.layers.Dense(
            units=output_units, activation=None)

        self.hidden_state = None
        self.cell_state = None

    def get_initial_state(self, batch_size):
        """
        Return a properly shaped initial state for the RNN.

        Args:
            batch_size: The size of the first dimension of the inputs
                        (excluding time steps).
        """
        hidden_state = tf.tile(self.initial_hidden_state[None, ...],
                               [batch_size, 1])
        cell_state = tf.tile(self.initial_cell_state[None, ...],
                             [batch_size, 1])
        return hidden_state, cell_state

    def call(self, states, actions, training=None, reset_state=None):
        """
        Generate predictions from the model.

        Args:
            states: The current states.
            actions: The current actions.
            training: Boolean to indicate training or inference.
                      Used for toggling dropout.
            reset_state: Boolean to reset the hidden state of the RNN.
        """
        inputs = tf.concat([states, actions], axis=-1)
        hidden = self.dense_embedding(inputs)

        # compute an initial state for the RNN
        if self.hidden_state is None or self.cell_state is None or reset_state:
            self.hidden_state, self.cell_state = self.get_initial_state(
                inputs.shape[0])

        hidden, self.hidden_state, self.cell_state = self.rnn(
            hidden, initial_state=[self.hidden_state, self.cell_state])
        logits = self.dense_logits(hidden)
        return logits
