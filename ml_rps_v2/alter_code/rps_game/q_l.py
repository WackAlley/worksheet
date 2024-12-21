my_dict = {}

#print(my_dict["bla"])
import numpy as np
array = np.array([[1, 4, -2, 3],[2, 3, 5, -2]])
print(array >= 4)  # Output: [False  True  True False]
#array[array >= 4] -= 3
#cond1 = np.logical_and(array >= 0, array <= 4)  # (array > 0) & (array < 4)


# np.select mit den Bedingungen
#result = array.copy()
#result[(array >= 1) & (array <= 3)] += 3
#result[array >= 4] -= 3
array = np.select(
            [(array >= 1) & (array <= 3), array >= 4],
            [array + 3, array  - 3],
            default=array)
x = np.array([[4,2,3], [7,0,5]])
idx = np.argmax(x)
idx_y, idx_x = np.unravel_index( np.argmax(x), x.shape)


print(idx)
print("Original Array:", x, idx_x, idx_y, x[idx_y, idx_x ] )
#print("Resulting Array:", result)
#np.select([(array>0) & (array<4), array>4], [array+3 , array-3],default= array)

#print(array)