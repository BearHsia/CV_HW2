import numpy as np
from nearest_neighbor_classify import nearest_neighbor_classify

train_feat = [[1,1],[2,1],[1,2],[2,2],
              [1,3],[2,3],[1,4],[2,4],
              [3,1],[4,1],[3,2],[4,2],
              [3,3],[4,3],[3,4],[4,4]]
train_feat = np.array(train_feat)
print(train_feat)
train_labe = ['circ','circ','circ','circ',
              'chec','chec','chec','chec',
              'tria','tria','tria','tria',
              'cros','cros','cros','cros']

test_feat = [[1.5,1.5],[3.5,1.5],[1.5,3.5],[3.5,3.5]]
test_feat = np.array(test_feat)
print(test_feat)

res = nearest_neighbor_classify(train_feat, train_labe, test_feat)
print(res)