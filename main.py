import numpy as np
import matplotlib.pyplot as plt



# classification models

from sklearn.tree import DecisionTreeClassifier, plot_tree




X = np.c_[(.4, -.7),
          (-1.5, -1),
          (-1.4, -.9),
          (-1.3, -1.2),
          (-1.1, -.2),
          (-1.2, -.4),
          (-.5, 1.2),
          (-1.5, 2.1),
          #--
          (1, 1),
          (1.3, .8),
          (1.2, .5),
          (.2, -2),
          (.5, -2.4),
          (.2, -2.3),
          (0, -2.7),
          (1.3, 2.1)].T
Y = [0] * 8 + [1] * 8

dt = DecisionTreeClassifier()
dt.fit(X,Y)
plt.show()
plot_tree(dt, filled=True)

dt = DecisionTreeClassifier(max_depth=2)
dt.fit(X, Y)
plt.show()
plot_tree(dt, filled=True);
dt.predict([[-0.8, -1]])
dt.predict([[1.1, 0]])
dt.predict([[0.5, -1]])
dt.predict([[0, 0]])