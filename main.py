import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

class NaiveGaussianBayes:
    def __init__(self):
        self.priors = None
        self.dist = None

    def fit(self, X, y):
        unique_labels = np.unique(y)
        counts = tf.unique_with_counts(labels).count

        # Compute priors
        self.priors = tf.cast(tf.math.log(counts / len(y)), tf.float32)

        # Separate data for each label
        data_per_label = tf.stack(
            [tf.concat(
                [tf.expand_dims(x, axis=0) for x,y in zip(X,y) if y==c],
                axis=0)
                for c in unique_labels
            ])

        # Compute means and variances
        means, variances = tf.nn.moments(data_per_label, axes=1)

        # Get Gaussian distribution
        self.dist = tfp.distributions.Normal(loc=means, scale=tf.sqrt(variances))

    def predict(self, X):
        assert self.dist is not None and self.priors is not None

        log_probs = self.priors + tf.math.reduce_sum(self.dist.log_prob(X), axis=2)

        return tf.math.argmax(log_probs, axis=0)

if __name__ == "__main__":
    nb = NaiveGaussianBayes()

    # Dataset
    iris = tfds.load(
        "iris",
        split="train",
        as_supervised=True
    )

    features = tf.concat([tf.expand_dims(x, axis=0) for x, y in iris], axis=0)
    labels = tf.concat([y for x, y in iris], axis=0)

    nb.fit(features, labels)





