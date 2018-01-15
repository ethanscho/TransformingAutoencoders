from data_loader import *
from transforming_autoencoder import *

X_trans, trans, X = load_data()
transforming_autoencoder = TransformingAutoencoder()

X_trans = np.reshape(X_trans, (-1, 28, 28, 1))
X = np.reshape(X, (-1, 28, 28, 1))

# for i in xrange(1000):
#     X_batch = X[0:100]
#     trans_batch = trans[0:100]
#     X_trans_batch = X_trans[0:100]

# print X_batch.shape, trans_batch.shape, X_trans_batch.shape 
transforming_autoencoder.model.fit([X, trans], X_trans, epochs=50, validation_split=0.1)
transforming_autoencoder.model.save('model.h5')

