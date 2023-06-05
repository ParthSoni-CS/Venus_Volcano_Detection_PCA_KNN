import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from skimage.transform import resize

def data_preprocess():
    """
    data_preprocessing() - use to import image cropping the labbled segment in an image and resizing it before appending it to X_train
    """
    X_train, Y_train = [], []
    # running loop for gathering samples
    for i in range(1,135):
        image_path = f'img{i}'

        #importing image and ground truth for that image
        img = np.fromfile(f'Images/{image_path}.sdt', dtype=np.uint8)
        information = np.loadtxt(f'Images/{image_path}.spr')
        lxyr = np.loadtxt(f'GroundTruths/{image_path}.lxyr')
        lxyr = np.array(lxyr)
        # color_distribution = {"1":"green", "2":"yellow", "3":"blue", "4":"red"}

        # fetching rows and cols from .spr
        nc = information[1]
        nr = information[4]

        #checking the dimensions 
        assert nc == 1024.00
        assert nr == 1024.00

        #reshaping the image based on dimension
        img = img.reshape((nr.astype(int),nc.astype(int)))
        # fig,ax = plt.subplots(1)
        # ax.imshow(img)

        #iterating throught each row of lxyr
        if(np.any(lxyr)):
            if len(lxyr.shape) == 2:
                for values in lxyr:
                    # fetching the label, x_center, y_center and radius for each detected segment 
                    label = values[0]
                    x_center = values[1]
                    y_center = values[2]
                    radius = values[3]
                    # circle = Circle((x_center, y_center),radius=radius, color = color_distribution[str(label.astype(int))])
                    # ax.add_patch(circle)

                    #cropping the segment using x_center, y_center and radius
                    cropped_image = img[np.ceil(y_center - radius).astype(int) if np.ceil(y_center - radius) >=0 else 0 : np.ceil(y_center + radius + 1).astype(int) if np.ceil(y_center + radius + 1) <= 1024 else 1024, np.ceil(x_center-radius).astype(int) if np.ceil(x_center-radius) >= 0 else  0 :np.ceil(x_center + radius + 1).astype(int) if np.ceil(x_center + radius + 1) <= 1024 else 1024]  
                    cropped_image_resized = resize(cropped_image, [40,40])

                    # ravel the cropped image and append the label, append them to X_train and Y_train repsectively for furthur process 
                    X_train.append(cropped_image_resized.ravel())
                    Y_train.append(label)        
            else:
                # fetching the label, x_center, y_center and radius for each detected segment 
                label = lxyr[0]
                x_center = lxyr[1]
                y_center = lxyr[2]
                radius = lxyr[3]

                #cropping the segment using x_center, y_center and radius
                cropped_image = img[np.ceil(y_center - radius).astype(int) if np.ceil(y_center - radius) >=0 else 0 : np.ceil(y_center + radius + 1).astype(int) if np.ceil(y_center + radius + 1) <= 1024 else 1024, np.ceil(x_center-radius).astype(int) if np.ceil(x_center-radius) >= 0 else  0 :np.ceil(x_center + radius + 1).astype(int) if np.ceil(x_center + radius + 1) <= 1024 else 1024]  
            

                # resizing the image so that every cropped image will have same number of feature
                cropped_image_resized = resize(cropped_image, [40,40])

                # ravel the cropped image and append the label, append them to X_train and Y_train repsectively for furthur process 
                X_train.append(cropped_image_resized.ravel())
                Y_train.append(label)

    # return X_train and Y_train as numpy array 
    return np.array(X_train), np.array(Y_train)

def PCA(X_train, num_components):
    # Calculate the mean of the input matrix X
    mean = np.mean(X_train, axis=0)
    # Center the input matrix X by subtracting the mean
    centered_X = X_train - mean
    # print(centered_X.shape)
    # Calculate the covariance matrix of the centered input matrix X
    covariance_matrix = np.dot(centered_X.T, centered_X)
    # print(covariance_matrix.shape)
    
    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
 
    #sorting on based of eigen values to get principal components
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    sorted_eigenvalues_real = sorted_eigenvalues.real

    # Take the top num_components eigenvectors
    principal_components = sorted_eigenvectors[:, :num_components]
    
    # Transform the centered input matrix X using the top eigenvectors to get the principal components
    transformed_X = np.dot(centered_X, principal_components)
    
    # Return the principal components and the explained variance
    explained_variance = sorted_eigenvalues_real[:num_components] / np.sum(sorted_eigenvalues_real)
    return transformed_X, explained_variance*100


def data_split_equal_by_class(X, seed = 1234, no_per_class= 6):
    """
    data_split_equal_by_class() - Use to split data by every class should have equal number in x_test 
    parameters - X - dataset that contain feature and its class
                 seed - fixing random state
                 split - splitting x_train and x_test 
    """

    # finding index on based of class value
    idx_1 = np.where(X[:,-1] == 1)[0]
    idx_2 = np.where(X[:,-1] == 2)[0]
    idx_3 = np.where(X[:,-1] == 3)[0]
    idx_4 = np.where(X[:,-1] == 4)[0]

    # assigning first n values of a particular class to test indexes  
    idx_1_test = idx_1[:no_per_class]
    idx_2_test = idx_2[:no_per_class]
    idx_3_test = idx_3[:no_per_class]
    idx_4_test = idx_4[:no_per_class]    

    # assigining remaining to train
    idx_1_train = idx_1[no_per_class:]
    idx_2_train = idx_2[no_per_class:]
    idx_3_train = idx_3[no_per_class:]
    idx_4_train = idx_4[no_per_class:]

    # using vstack to append test and train data with respective x_test and x_train
    data_train = np.vstack((X[idx_1_train], X[idx_2_train], X[idx_3_train], X[idx_4_train]))
    data_test = np.vstack((X[idx_1_test], X[idx_2_test], X[idx_3_test], X[idx_4_test]))

    #shuffling test and train data 
    np.random.seed(seed)
    np.random.shuffle(data_train)

    np.random.seed(seed)
    np.random.shuffle(data_test)

    #getting x_train, y_train, x_test and y_test 
    X_train = data_train[:,:-1].astype(float)
    Y_train = data_train[:,-1]
    X_test = data_test[:,:-1].astype(float)
    Y_test = data_test[:,-1]
    return X_train, Y_train, X_test, Y_test

def data_split_normal(X, split = 0.75):
    """
    data_split_normal() - splitting the data based on normal formula for data split
    parameters - X - Dataset that contains X_train with each row having it's respective value 
               - split - split percentage for X_train and X_test 
    """
    #shuffle data after setting the seed
    np.random.seed(42)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    # Split the indices into train and test sets
    train_indices = indices[:int(split * X.shape[0])]
    test_indices = indices[int(split * X.shape[0]):]
    data_train = X[train_indices]
    data_test = X[test_indices]

    #Get X_train, Y_train, X_test and Y_test from data_train and data_test 
    X_train = data_train[:,:-1]
    Y_train = data_train[:,-1]
    X_test =  data_test[:,:-1]
    Y_test = data_test[:,-1]

    return X_train, Y_train, X_test, Y_test


def data_split(X_train, Y_train, split_equal_by_class = False):
    """
    data_split() - split data based on either by equal number of class in X_test or by normal method
    parameters - X_train - Training data 
               - Y_train - Training label 
               - split_equal_by_class - boolean variable if true x_test will have all the classes having same number of sample
    """

    # split data based on either by equal number of class in X_test or by normal method
    no_per_class = 6
    Y_train = Y_train.reshape(-1,1)
    data = np.append(X_train, Y_train, axis = 1)
    if split_equal_by_class == True:
        X_train, Y_train, x_test, y_test = data_split_normal(data, no_per_class)
    else:
        X_train, Y_train, x_test, y_test = data_split_normal(data, 0.75)
    
    return X_train, Y_train, x_test, y_test

def cartesianDistance(p1, p2):
    """
    cartesianDistance() - Used to calculate cartesian distance between
                          two points of same dimensions.
    parameters - p1 - point 1.
               - p2 - point 2.
    """
    distance = 0.00
    for i, j in zip(p1, p2):
        distance = distance + (i-j)**2
    return distance**(0.5)


def manhattanDistance(p1, p2):
    """
    manhattanDistance() - Used to calculate manhattan distance between
                          two points of same dimensions.
    parameters - p1 - point 1.
               - p2 - point 2.
    """
    distance = 0.00
    for i, j in zip(p1, p2):
        distance = distance + abs(i-j)
    return distance

def knn_predict(dataset, input, k=3, distanceMeasure="cartesian", distance_weight = False, class_weight = False):
    """
    predict() - Used to predict the class of a input point using KNN. 
    parameters - dataset - train data used for KNN distance measurement.
               - input - point of which class is predicted.
               - k - k value used in knn. 
               - distanceMeasure - the metric used to calculate distance between two points
                                   ,two choices are available, cartesian and manhattan. default is 
                                   cartesian.
    """
    distance = []
    labels = np.array(dataset[:,-1]).reshape(-1,1)
    # print(np.sum(labels == 1), np.sum(labels == 2), np.sum(labels == 3), np.sum(labels == 4))
    #calculating the distance of each point of X_train w.r.t to input point and appending it and label of the point to distance list 
    for point in dataset:
        distance.append(((manhattanDistance(point[0:-1], input) if distanceMeasure == "manhattan" else cartesianDistance(point[0:-1], input)), point[-1]))
    
    #sort distance on ascending order
    distance.sort(key = lambda a:a[0])

    #get the first k element of distace 
    distance = distance[0:k]

    #finding probability for each label
    probability = dict()
    for d in distance:
        if d[1] in probability.keys():
            if distance_weight and class_weight:
                probability[d[1]] = probability[d[1]] + 1/(d[0]*np.sum(labels == d[1]))
            elif distance_weight and not class_weight:
                probability[d[1]] = probability[d[1]] + 1/(d[0])
            elif class_weight and not distance_weight:
                probability[d[1]] = probability[d[1]] + 1/np.sum(labels == d[1])
            else:
                # print("In this part ")
                probability[d[1]] = probability[d[1]] + 1

        else:
            if distance_weight and class_weight:
                probability[d[1]] = 1/(d[0]*np.sum(labels == d[1]))
            elif distance_weight and not class_weight:
                probability[d[1]] = 1/(d[0])
            elif not distance_weight and class_weight:
                probability[d[1]] = 1/np.sum(labels == d[1])
            else:
                probability[d[1]] = 1
    v = list(probability.values())
    k = list(probability.keys())

    # returning the class which has the highest probability
    predictedClass = k[v.index(max(v))]
    return predictedClass

def knn_classifier(X_train,Y_train, x_test, y_test, k = 3, distance = "cartesian", bagging = False, number_of_classifier = None, distance_weight = False, class_weight = False):
    if bagging == False:
        Y_train = Y_train.reshape(-1,1)
        dataset = np.append(X_train, Y_train, axis = 1)
        point = x_test[3]
        y_pred = []
        for i in range(x_test.shape[0]):
            label = knn_predict(dataset=dataset, input = x_test[i], k = k, distanceMeasure = distance, distance_weight=distance_weight, class_weight=class_weight)
            y_pred.append(label)
        y_test = y_test.reshape(-1,1)
        y_pred = np.array(y_pred)
        y_pred = y_pred.reshape(-1,1)
        # print(np.mean(y_pred==y_test))
        return y_pred
    else:
        for classifier_number in [number_of_classifier]:
            result = np.full_like(x_test[:,-1], 0.0)
            result = result.reshape(-1,1)
            Y_train = Y_train.reshape(-1,1)
            data = np.append(X_train, Y_train, axis = 1)
            idx = [i for i in range(data.shape[0])]
            idx_random = []
            for i in range(classifier_number):
                y_pred = []
                idx_random.append(np.random.choice(idx[:], data.shape[0]))
                data_random = []
                for c in idx_random[i]:
                    data_random.append(list(data[c,:]))
                data_random = np.array(data_random)
                X_train = np.array(data_random[:,:-1], dtype=np.float64)
                Y_train = np.array(data_random[:,-1]).reshape(-1,1)
                for j in range(x_test.shape[0]):
                    label = knn_predict(dataset=data_random, input = x_test[j], k = k, distanceMeasure = distance, distance_weight=distance_weight, class_weight=class_weight)
                    y_pred.append(label)
                result = np.hstack((result, np.array(y_pred).reshape(-1,1)))
            result = result[:,1:]
            y_pred_ensemble = []
            for i in range(result.shape[0]):   
                value, counts = np.unique(result[i,:], return_counts = True)
                y_pred_ensemble.append(value[np.argmax(counts)])
            y_pred_ensemble = np.array(y_pred_ensemble).reshape(-1,1)
        return y_pred_ensemble

def class_wise_accuracy(y_pred, y_test):
    unique_labels = np.unique(y_test)
    accuracy_results= []
    for label in unique_labels:
        y_test_label = (y_test == label)
        y_pred_label = y_pred[y_test_label]
        accuracy = np.mean(y_pred_label == label)
        accuracy_results.append(accuracy)
    return accuracy_results

def main():
    X_train, Y_train = data_preprocess()

    # Define range of PCA components and k values to test
    
    pca_components = [2, 3, 5, 10, 20, 30, 40, 80, 100, 120, 150]
    k_values = [1, 2, 3, 5, 7, 9, 11, 15]


    accuracy_results = {}

    # Loop through all combinations of PCA components and k values
    for pca in pca_components:
        for k in k_values:
            # Preprocess data and perform PCA
            X_train, Y_train = data_preprocess()
            pc, var = PCA(X_train, pca)
            X_train_pc, Y_train_pc, x_test_pc, y_test_pc = data_split(pc.real, Y_train)
            # Perform KNN classification with current parameters
            y_pred = knn_classifier(X_train_pc, Y_train_pc, x_test_pc, y_test_pc, bagging=False, k=k)
            # Calculate accuracy and store in dictionary
            accuracy = np.mean(y_pred == y_test_pc.reshape(-1,1))
            accuracy_results[(pca, k)] = accuracy*100

    # Print accuracy results in a grid
    print("Test Accuracy results:")
    print("PCA", end="\t")
    for k in k_values:
        print("k={}".format(k), end="\t")
    print()
    for pca in pca_components:
        print(pca, end="\t")
        for k in k_values:
            accuracy = accuracy_results[(pca, k)]
            print("{:.3f}".format(accuracy), end="\t")
        print()

    
    accuracy_results = {}

    # Loop through all combinations of PCA components and k values
    for pca in pca_components:
        for k in k_values:
            # Preprocess data and perform PCA
            X_train, Y_train = data_preprocess()
            pc, var = PCA(X_train, pca)
            X_train_pc, Y_train_pc, x_test_pc, y_test_pc = data_split(pc.real, Y_train)
            # Perform KNN classification with current parameters
            y_pred = knn_classifier(X_train_pc, Y_train_pc, x_test_pc, y_test_pc, bagging=False, k=k, distance_weight=True)
            # Calculate accuracy and store in dictionary
            accuracy = np.mean(y_pred == y_test_pc.reshape(-1,1))
            accuracy_results[(pca, k)] = accuracy*100

    # Print accuracy results in a grid
    print("Test Accuracy results using distance as a weight for calculating probability in knn:")
    print("PCA", end="\t")
    for k in k_values:
        print("k={}".format(k), end="\t")
    print()
    for pca in pca_components:
        print(pca, end="\t")
        for k in k_values:
            accuracy = accuracy_results[(pca, k)]
            print("{:.3f}".format(accuracy), end="\t")
        print()

    
    accuracy_results = {}

    # Loop through all combinations of PCA components and k values
    for pca in pca_components:
        for k in k_values:
            # Preprocess data and perform PCA
            X_train, Y_train = data_preprocess()
            pc, var = PCA(X_train, pca)
            X_train_pc, Y_train_pc, x_test_pc, y_test_pc = data_split(pc.real, Y_train)
            # Perform KNN classification with current parameters
            y_pred = knn_classifier(X_train_pc, Y_train_pc, x_test_pc, y_test_pc, bagging=False, k=k, class_weight=True)
            # Calculate accuracy and store in dictionary
            accuracy = np.mean(y_pred == y_test_pc.reshape(-1,1))
            accuracy_results[(pca, k)] = accuracy*100

    # Print accuracy results in a grid
    print("Test Accuracy results using number of class as a weight for calculating probability in knn:")
    print("PCA", end="\t")
    for k in k_values:
        print("k={}".format(k), end="\t")
    print()
    for pca in pca_components:
        print(pca, end="\t")
        for k in k_values:
            accuracy = accuracy_results[(pca, k)]
            print("{:.3f}".format(accuracy), end="\t")
        print()

    
    for pca in [2,5,10,15,20,30,60,90,120,150,200]:
        print(f"Applying for PCA = {pca}")
        accuracy_results_entire_k = []
        k_values = [2,3,5,7,9,11,13,15]
        for k in k_values:
            X_train, Y_train = data_preprocess()
            pc, var = PCA(X_train, pca)
            X_train_pc, Y_train_pc, x_test_pc, y_test_pc = data_split(pc.real, Y_train)
            y_pred = knn_classifier(X_train_pc, Y_train_pc, x_test_pc, y_test_pc, bagging=False, k=k, class_weight=True)
            accuracy_results = class_wise_accuracy(y_pred, y_test_pc)
            accuracy_results_entire_k.append(accuracy_results)
            # Print class-wise accuracy for all k values in tabular form
        print("K\t", end="")
        for label in np.unique(Y_train):
            print("Class {}\t".format(label), end="")
        print()
        for i in range(len(k_values)):
            print("{}\t".format(k_values[i]), end="")
            for label in np.unique(Y_train):
                accuracy = accuracy_results_entire_k[i][label.astype(int)-1]
                print("{:.3f}\t\t".format(accuracy), end="")
            print()



    
if __name__ == "__main__":
    main()