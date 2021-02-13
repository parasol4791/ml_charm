# Utilities for plotting results
from matplotlib import pyplot as plt


def smooth_curve(points, factor=0.9):
    """Smoothes a curve by computing a weighted moving average"""
    smooth_points = []
    for p in points:
        if smooth_points:
            prev = smooth_points[-1]
            smooth_points.append(prev*factor + p*(1-factor))
        else:
            smooth_points.append(p)
    return smooth_points


def plot_the_model(trained_weight, trained_bias, feature, label):
    """Plots the trained model against the training feature and label"""
    # Label the axes.
    plt.xlabel("feature")
    plt.ylabel("label")

    # Plot the feature values vs. label values.
    plt.scatter(feature, label)

    # Create a red line representing the model. The red line starts
    # at coordinates (x0, y0) and ends at coordinates (x1, y1).
    x0 = 0
    y0 = trained_bias
    x1 = feature[-1]
    y1 = trained_bias + (trained_weight * x1)
    plt.plot([x0, x1], [float(y0), float(y1)], c='r')

    # Render the scatter plot and the red line.
    plt.show()


def plot_the_loss_curve(epochs, rmse):
    """Plot the loss curve, which shows loss vs. epoch"""
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs, rmse, label="Loss")
    plt.legend()
    plt.ylim([rmse.min()*0.97, rmse.max()])
    plt.show()


def plot_accuracy_loss(
    train_acc,
    train_loss,
    valid_acc,
    valid_loss
):
    """Plots 2 graphs - for trained/validation accuracy and loss"""
    epochs = range(1, len(train_acc) + 1)
    plt.plot(epochs, train_acc, 'bo', label='Training acc')
    plt.plot(epochs, valid_acc, 'b', label='Validation acc')
    plt.title('Training & validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.figure()
    plt.plot(epochs, train_loss, 'bo', label='Training loss')
    plt.plot(epochs, valid_loss, 'b', label='Validation loss')
    plt.title('Training & validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
