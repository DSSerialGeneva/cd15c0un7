import numpy as np
# c = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
c = np.array([[1, 2, 3], [4, 2, 6], [7, 8, 9]])
# c = dict()
# c[(1, 2)] = 3
# c[(4, 2)] = 6
# c[(7, 8)] = 9


# d = np.empty_like([0, 1, 2])

# d = np.array([1, 2, 3])
# print(d)
#
# d = np.vstack((d, [0, 1, 2]))
#
# print(d)
#
# print(d[0, 1])

#
category_id = 4
#
md52 = 2
#
# print(c)
same_pictures = np.nonzero(c[:, 1] == md52)[0]
# not_new_picture = len(same_pictures) > 0
# is_new_category = True
print(same_pictures)
# if is_new_category and not_new_picture:
#     for i_same_picture in same_pictures:
#         print("i_same_picture %s" % i_same_picture)
#         print(c[i_same_picture, 0])
#         same_category = c[i_same_picture, 0] == category_id
#         print(same_category)
#         if same_category:
#             is_new_category = False
#             break
# else:
#     is_new_category = False  # because we don't want to mark the category as new when the picture is new too
# print("Final is new category ? %s" % is_new_category)
