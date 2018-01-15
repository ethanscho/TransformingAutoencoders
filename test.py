from data_loader import *
from transforming_autoencoder import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

X_trans, trans, X = load_test_data()
transforming_autoencoder = TransformingAutoencoder()

X_trans = np.reshape(X_trans, (-1, 28, 28, 1))
X = np.reshape(X, (-1, 28, 28, 1))

result = transforming_autoencoder.model.predict([X, trans])

""" Plot """
X = np.reshape(X, (-1, 28, 28))
X_trans = np.reshape(X_trans, (-1, 28, 28))
result = np.reshape(result, (-1, 28, 28))

gs = gridspec.GridSpec(6, 9, top=1., bottom=0., right=1., left=0., hspace=0., wspace=0.)

for i, g in enumerate(gs):
    ax = plt.subplot(g)

    if i % 3 == 0:
        ax.imshow(result[i / 3], vmin=0, vmax=1)
    elif i % 3 == 1:
        ax.imshow(X_trans[i / 3], vmin=0, vmax=1)
    else:
        ax.imshow(X[i / 3], vmin=0, vmax=1)
        
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()


