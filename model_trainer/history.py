from matplotlib.pyplot as plt


def plot_history_loss(history):
    epochs = range(1, len(history['val_loss'])+1) 
    plt.figure(figsize=(18, 6))

    plt.subplot(131)
    plt.plot(epochs, history['loss'], label="train loss")
    plt.plot(epochs, history['val_loss'], label="valid loss")
    plt.legend()
    plt.subplot(132)
    plt.plot(epochs, history['mrcnn_class_loss'], label="train class loss")
    plt.plot(epochs, history['val_mrcnn_class_loss'], label="valid class loss")
    plt.legend()
    plt.subplot(133)
    plt.plot(epochs, history['mrcnn_mask_loss'], label="train mask loss")
    plt.plot(epochs, history['val_mrcnn_mask_loss'], label="valid mask loss")
    plt.legend()

    plt.show()