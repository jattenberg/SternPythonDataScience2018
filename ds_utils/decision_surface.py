import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def Decision_Surface(data, col1, col2, target, model, probabilities=False, gridsize=0.5, sample=1):
    # Get bounds
    x_min, x_max = data[col1].min(), data[col1].max()
    y_min, y_max = data[col2].min(), data[col2].max()
    
    # Create a mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, gridsize), np.arange(y_min, y_max, gridsize))
    meshed_data = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])
    
    tdf = data[[col1, col2]]

    plt.ylabel(col2)
    plt.xlabel(col1)

    if None != model:
        model.fit(tdf, target)
        if probabilities:
            # Color-scale on the contour (surface = separator)
            Z = model.predict_proba(meshed_data)[:, 1].reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.4)
        else:
            # Only a curve/line on the contour (surface = separator)
            Z = model.predict(meshed_data).reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.4)
    
    color = ["blue" if t == 0 else "red" for t in target]
    indices = np.random.permutation(range(len(color)))[:int(sample*len(color))].tolist()

    plt.scatter(data[col1][indices], data[col2][indices], color=np.array(color)[indices] )



def Regression_Surface(data, col1, col2, target, model, gridsize=0.5, sample=1):
    # Get bounds
    x_min, x_max = data[col1].min(), data[col1].max()
    y_min, y_max = data[col2].min(), data[col2].max()
    
    # Create a mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, gridsize), np.arange(y_min, y_max, gridsize))
    meshed_data = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])
    
    tdf = data[[col1, col2]]

    if model:
        model.fit(tdf, target)
        Z = model.predict(meshed_data).reshape(xx.shape)

    plt.ylabel(col2)
    plt.xlabel(col1)
    
    cs = plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.coolwarm)

    length = target.count()
    indices = np.random.permutation(range(length))[:int(sample*length)]

    plt.scatter(data[col1][indices], data[col2][indices], c=target[indices], cmap=plt.cm.coolwarm )
