from PIL import Image
import numpy as np

img = Image.open("../hw2_data/p1/train/Bedroom/image_0001.jpg")
img = img.resize((20, 20))
output_matrix = np.array(img)
output_matrix = np.reshape(output_matrix,(-1,400))
print(output_matrix.shape)

test_a = [[9,2,1],[11,10,12],[99,100,101]]
test_a = np.array(test_a)
print(test_a)

avg_of_rows = test_a.sum(axis=1)/3
print(avg_of_rows)
test_a = test_a - avg_of_rows[:,np.newaxis]
print(test_a)

#test_tmp = test_a*test_a
#print(test_tmp)
sqrt_sum_of_rows = np.sqrt(np.square(test_a).sum(axis=1))
print(sqrt_sum_of_rows)
test_a = test_a/sqrt_sum_of_rows[:,np.newaxis]
print(test_a)

print("-----------------------")
index_a = np.argsort(test_a, axis=0)
print(index_a)

cls_list = ['a','b','c']

for col_i in range(index_a.shape[1]):
    col_1_index = index_a[:,col_i].tolist()
    #print(col_1_index)
    cls_idx_zip = zip(cls_list,col_1_index)
    #print(cls_idx_zip)
    sorted_cls_list = sorted(cls_idx_zip, key=lambda x:x[1])
    sorted_cls_list,_ = zip(*sorted_cls_list)
    sorted_cls_list = list(sorted_cls_list)
    print(sorted_cls_list)

print("-----------------------")
test_dic = {}
test_dic['c'] = 1
test_dic['a'] = 2
test_dic['b'] = 3
print(max(test_dic, key=(lambda key: test_dic[key])))