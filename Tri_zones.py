from Projet import *
import matplotlib.image as mpimg
Liste_indices = [[0 for i in range(NUM_CLASSES + 1)] for j in range(3)]

Liste_labels_classe = np.full(2000, -2)

for i in range(2000):
    plt.ion()
    img = plt.imshow(data_train[i])
    plt.axis("off")
    plt.show()

    label = input("Classe de l'image :")
    while len(label) > 1 :
        label = input("Classe de l'image")
    
    Liste_labels_classe[i] = int(label)


plt.ioff()






