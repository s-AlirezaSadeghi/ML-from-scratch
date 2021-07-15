from matplotlib import pyplot as plt

def plot_loss(trained_model):
    loss = [x[0] for x in trained_model.rsme_tracker]
    plt.figure()
    plt.plot(range(0, len(trained_model.rsme_tracker)), loss)
    plt.show()