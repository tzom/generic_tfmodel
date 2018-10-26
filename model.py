import tensorflow as tf
from scope_decorator import define_scope
from tensorflow.examples.tutorials.mnist import input_data as iptf


class Model:

    def __init__(self, batch):
        self.learning_rate = 0.01
        self.X ,self.label = batch
        self._prediction
        self._create_loss_optimizer
        #self.optimize
        #self.error
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def _prediction(self):
        x = self.X
        x = tf.contrib.slim.fully_connected(x, 200)
        x = tf.contrib.slim.fully_connected(x, 200)
        x = tf.contrib.slim.fully_connected(x, 10, tf.nn.softmax)
        return x

    @define_scope
    def _create_loss_optimizer(self):
        logprob = tf.log(self._prediction + 1e-12)
        cross_entropy = -tf.reduce_sum(self.label * logprob)
        self.loss = cross_entropy
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def partial_fit(self, batch):
        """Train model based on mini-batch of input data.        
        Return loss of mini-batch.
        """
        X, label = batch
        opt, loss = self.sess.run((self.optimizer, self.loss), 
                                  feed_dict={self.X: X, self.label: label})
        return loss

    def predict(self, batch):
        """Train model based on mini-batch of input data.        
        Return loss of mini-batch.
        """
        X, label = batch
        y = self.sess.run(tf.argmax(self._prediction,1), 
                                  feed_dict={self.X: X, self.label: label})
        return y

def train(data):

    training_epochs = 10
    n_samples = 100
    batch_size = 10 
    display_step = 2

    X = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.float32, [None, 10])
    batch = [X,label]
    model = Model(batch)

    
    
    for epoch in range(training_epochs):
        avg_loss = 0.
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            # Fit training using batch data
            batch = data.train.next_batch(100)
            loss = model.partial_fit(batch)
            # Compute average loss
        avg_loss += loss / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), 
                  "loss=", "{:.9f}".format(avg_loss))
    return model

def main():
    mnist = iptf.read_data_sets("../mnist/", one_hot=True)
    model = train(mnist)

    batch = mnist.train.next_batch(100)
    prediction = model.predict(batch)
    print(prediction)


if __name__ == '__main__':
    main()