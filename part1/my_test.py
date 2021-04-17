from PIL import Image
import numpy as np

img = Image.open("../hw2_data/p1/train/Bedroom/image_0001.jpg")
img = img.resize((20, 20))
output_matrix = np.array(img)
output_matrix = np.reshape(output_matrix,(-1,400))
print(output_matrix.shape)

test_a = [[1,2,9],[10,11,12],[99,100,101]]
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