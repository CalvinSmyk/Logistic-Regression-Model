import os
import matplotlib.pyplot as plt
from LogisticRegression import logistic_regression
from DataReader import *

data_dir = "../data/"
train_filename = "training.npz"
test_filename = "test.npz"
    
def visualize_features(X, y):
    '''This function is used to plot a 2-D scatter plot of training features.

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.

    Returns:
        No return. Save the plot to 'train_features.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    plt.scatter(X[:,0],X[:,1],c=y)
    plt.title('Scatter Plot')
    plt.xlabel('Feature_symmetry')
    plt.ylabel('Feature_intensity')
    plt.savefig('feature_separation')

    ### END YOUR CODE

def visualize_result(X, y, W):
    """This function is used to plot the sigmoid model after training.

	Args:
		X: An array of shape [n_samples, 2].
		y: An array of shape [n_samples,]. Only contains 1 or -1.
		W: An array of shape [n_features,].

	Returns:
		No return. Save the plot to 'train_result_sigmoid.*' and include it
		in submission."""

    ### YOUR CODE HERE
    """w1,w2 = W[1],W[2]
    slope = -w1/w2

    preds = []
    for x in X:
        prediction = 1 / (1 + np.exp(-1 * (np.dot(x, W))))
        prediction = 1 if (prediction > 0.5) else -1
        preds.append(prediction)
    plt.plot(X,preds, c='lightblue', linewidth=3.0)
    plt.scatter(
        X[(y==1).ravel()],
        y[(y==1).ravel()],
        marker=".",
        linewidth=1.0,
        label="passed",
    )
    plt.scatter(
        X[(y == -1).ravel()],
        y[(y == -1).ravel()],
        marker=".",
        linewidth=1.0,
        label="passed",
    )
    plt.axhline(y=0.5,color ="orange", linestayle ="--", label="boundary")
    plt.xlabel("Iterations spend learning")
    plt.ylabel('p("passing the test")')
    plt.legend(frameon = False, loc="best", bbox_to_anchor=(0.5,0.0,0.5,0.5), prop={'size':20})
    plt.show()"""
############
    ### END YOUR CODE
    """min1,max1 = X[:,0].min()-1, X[:,0].max()+1
    min2,max2 = X[:,1].min()-1, X[:,1].max()+1
    x1grid = np.arange(min1,max1,0.1)
    x2grid = np.arange(min2,max2,0.1)

    xx, yy = np.meshgrid(x1grid,x2grid)

    r1,r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1),1)), r2.reshape((len(r2),1))

    grid = np.hstack((r1,r2))
    preds = []
    for x in grid:
        prediction = 1 / (1 + np.exp(-1 * (np.dot(x, W))))
        prediction = 1 if (prediction > 0.5) else -1
        preds.append(prediction)

    zz = preds.reshape(xx.shape)
    plt.contourf(xx,yy,zz,cmap='Paired')

    for class_value in range([-1,1]):
        row_ix = np.where(y == class_value)
        plt.scatter(X[row_ix,-1], X[row_ix, 1], cmap='Paired')"""
#############
    """x_set, y_set = X, y
    x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))

    # Ploting
    input = np.array([x1.ravel(), x2.ravel()]).T
    function = 1 / (1 + np.exp(-1 * (np.dot(input, W[1:3]))))
    function = 1 if function > 0.5 else -1
    output = function.reshape(x1.shape)
    plt.contourf(x1, x2,output.reshape(x1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'yellow')))
    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())

    # for loop to iterate the data
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                    c=ListedColormap(('black', 'green'))(i), label=j)

        # labeling the graph
    plt.title('Logistic Regression (Training set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()"""
###############
    b = W[0]
    w1,w2 = W[1],W[2]
    c = -b/w2
    m = -w1/w2

    x_axis_min, x_axis_max = X[:,0].min()-0.25, X[:,0].max()+0.25
    y_axis_min, y_axis_max = X[:, 1].min() - 0.25, X[:, 1].max() + 0.25

    xd = np.array([x_axis_min,x_axis_max])
    yd = m*xd + c
    plt.plot(xd, yd, 'k', lw=1, ls='--')
    plt.fill_between(xd, yd, y_axis_min, color='tab:blue', alpha=0.2)
    plt.fill_between(xd, yd, y_axis_max, color='tab:orange', alpha=0.2)

    plt.scatter(*X[y == 0].T, s=8, alpha=0.5)
    plt.scatter(*X[y == 1].T, s=8, alpha=0.5)
    plt.xlim(x_axis_min, x_axis_max)
    plt.ylim(y_axis_min, y_axis_max)
    plt.ylabel(r'$x_2$')
    plt.xlabel(r'$x_1$')
    plt.savefig('train_result_sigmoid')



def visualize_result_multi(X, y, W):
    '''This function is used to plot the softmax model after training.

	Args:
		X: An array of shape [n_samples, 2].
		y: An array of shape [n_samples,]. Only contains 0,1,2.
		W: An array of shape [n_features, 3].
	
	Returns:
		No return. Save the plot to 'train_result_softmax.*' and include it
		in submission.'''
    ### YOUR CODE HERE

	### END YOUR CODE

def main():
    # ------------Data Preprocessing------------
	# Read data for training.
    
    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)

    ##### Preprocess raw data to extract features
    train_X_all = prepare_X(raw_train)
    valid_X_all = prepare_X(raw_valid)
    ##### Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
    train_y_all, train_idx = prepare_y(label_train)
    valid_y_all, val_idx = prepare_y(label_valid)  

    ####### For binary case, only use data from '1' and '2'  
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    ####### Only use the first 1350 data examples for binary training. 
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    ####### set lables to  1 and -1. Here convert label '2' to '-1' which means we treat data '1' as postitive class. 
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1
    data_shape= train_y.shape[0]

#    # Visualize training data.
    visualize_features(train_X[:, 1:3], train_y)


   # ------------Logistic Regression Sigmoid Case------------

   ##### Check BGD, SGD, miniBGD
    logisticR_classifier = logistic_regression(learning_rate=0.1, max_iter=1000)

    logisticR_classifier.fit_miniBGD(train_X, train_y, data_shape)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_SGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_BGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 1)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))


    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    """
    tried to alternate learning rate [0.01,0.1,0.2,0.3,0.4,0.5]
    tried to alternate iterations [100,500,1000,3000]
    tried to alternate batch size [2,4,8,16,32]"""
    ### END YOUR CODE

    # Visualize your 'best' model after training.
    ### YOUR CODE HERE
    best_logisticR = logistic_regression(learning_rate=0.1, max_iter=1000)
    best_logisticR = best_logisticR.fit_miniBGD(train_X, train_y,batch_size=1)
    visualize_result(train_X[:, 1:3], train_y, best_logisticR.get_params())
    ### END YOUR CODE

    # Use the 'best' model above to do testing. Note that the test data should be loaded and processed in the same way as the training data.
    ### YOUR CODE HERE

    test_data,labels= load_data(os.path.join(data_dir, test_filename))

    ##### Preprocess raw data to extract features
    test_X_all = prepare_X(test_data)
    ##### Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
    test_y_all, test_idx = prepare_y(labels)

    ####### For binary case, only use data from '1' and '2'
    test_X = test_X_all[test_idx]
    test_y = test_y_all[test_idx]
    ####### Only use the first 1350 data examples for binary training.
    test_X = test_X[0:1350]
    test_y = test_y[0:1350]
    ####### set lables to  1 and -1. Here convert label '2' to '-1' which means we treat data '1' as postitive class.
    test_y[np.where(test_y == 2)] = -1
    data_shape = test_y.shape[0]

    print(f'final accuracy of best Model: {best_logisticR.score(test_X, test_y)}')


    ### END YOUR CODE



    # ------------End------------
    

if __name__ == '__main__':
	main()
    
    
