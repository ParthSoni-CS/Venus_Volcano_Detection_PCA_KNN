import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from process import data_preprocess, PCA
def display_image_with_patches(image_number = 2):
    image_path = f'img{image_number}'
    img = np.fromfile(f'Images/{image_path}.sdt', dtype=np.uint8)
    information = np.loadtxt(f'Images/{image_path}.spr')
    lxyr = np.loadtxt(f'GroundTruths/{image_path}.lxyr')
    lxyr = np.array(lxyr)
    color_distribution = {"1":"green", "2":"yellow", "3":"blue", "4":"red"}
    nc = information[1]
    nr = information[4]
    assert nc == 1024.00
    assert nr == 1024.00
    img = img.reshape((nr.astype(int),nc.astype(int)))
    fig,ax = plt.subplots(1)
    ax.imshow(img)
    if(np.any(lxyr)):
        if len(lxyr.shape) == 2:
            for values in lxyr:
                label = values[0]
                x_center = values[1]
                y_center = values[2]
                radius = values[3]
                circle = Circle((x_center, y_center),radius=radius, color = color_distribution[str(label.astype(int))])
                ax.add_patch(circle)
        else:
            label = lxyr[0]
            x_center = lxyr[1]
            y_center = lxyr[2]
            radius = lxyr[3]
            cropped_image = img[np.ceil(y_center - radius).astype(int) if np.ceil(y_center - radius) >=0 else 0 : np.ceil(y_center + radius + 1).astype(int) if np.ceil(y_center + radius + 1) <= 1024 else 1024, np.ceil(x_center-radius).astype(int) if np.ceil(x_center-radius) >= 0 else  0 :np.ceil(x_center + radius + 1).astype(int) if np.ceil(x_center + radius + 1) <= 1024 else 1024]  
            print(cropped_image.shape)  
            circle = Circle((x_center, y_center),radius=radius, color = color_distribution[str(label.astype(int))])
            ax.add_patch(circle)  
    plt.show() 

def display_radii_distribution():
    radii_probably_volcano = []
    radii_possibly_volcano = []
    radii_definately_volcano = []
    radii_pit = []
    for i in range(1,135):
        image_path = f'img{i}'

        #importing image and ground truth for that image
        
        lxyr = np.loadtxt(f'GroundTruths/{image_path}.lxyr')
        if(np.any(lxyr)):
            if len(lxyr.shape) == 2:
                for values in lxyr:
                    label = values[0]
                    x_center = values[1]
                    y_center = values[2]
                    radius = values[3]

                    if label == 1:
                        radii_definately_volcano.append(radius)
                    elif label == 2:
                        radii_probably_volcano.append(radius)
                    elif label == 3:
                        radii_possibly_volcano.append(radius)
                    else:
                        radii_pit.append(radius)
            else:
                label = lxyr[0]
                x_center = lxyr[1]
                y_center = lxyr[2]
                radius = lxyr[3]

                if label == 1:
                    radii_definately_volcano.append(radius)
                elif label == 2:
                    radii_probably_volcano.append(radius)
                elif label == 3:
                    radii_possibly_volcano.append(radius)
                else:
                    radii_pit.append(radius)

    bins =  [i for i in range(0,105, 5)]
    fig, ax = plt.subplots(figsize =(10, 7))
    ax.hist(np.array(radii_definately_volcano), bins =bins, color = "green")
    ax.set_ylabel("No of definately volcano")
    ax.set_xlabel("Radius of volcano")
    plt.show()

    fig, ax = plt.subplots(figsize =(10, 7))
    ax.hist(np.array(radii_probably_volcano), bins =bins, color = "yellow")
    ax.set_ylabel("No of probably volcano")
    ax.set_xlabel("Radius of volcano")
    plt.show()

    fig, ax = plt.subplots(figsize =(10, 7))
    ax.hist(np.array(radii_possibly_volcano), bins =bins, color = "blue")
    ax.set_ylabel("No of possibly volcano")
    ax.set_xlabel("Radius of volcano")
    plt.show()

    fig, ax = plt.subplots(figsize =(10, 7))
    ax.hist(np.array(radii_pit), bins =bins, color = "red")
    plt.show()

def display_pca():
    ls_var = np.array([])
    X1 = np.array([])
    X_train, Y_train = data_preprocess()
    pc, var = PCA(X_train, 250)
    print(var[:200])
    for x in [2,5, 15,25,35,45,70,200]:
        s_var = np.sum(var[:x])
        ls_var =np.append(ls_var,s_var)
        X1 =np.append(X1,x)
        #print(ls_var*100)
    plt.plot(X1, ls_var, 'go--', linewidth = 2, markersize=12)
    plt.xlabel('Number of principal components')
    plt.ylabel('Percent variance explained')
    plt.show()

def display_imbalance():
    X_train, Y_train = data_preprocess()
    value, count = np.unique(Y_train, return_counts = True)
    plt.bar(value, count)
    plt.xlabel("Labels")
    plt.ylabel("No of data")
display_pca()
display_imbalance()
display_image_with_patches()
display_radii_distribution()
