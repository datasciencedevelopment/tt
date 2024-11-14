# from Math_Functions import SEO_Math
#
# x = 5
# y = 6
# result = SEO_Math.testmath(x, y)
# print(result)



# n = int(input())
#
# simple_list = [1, 2]
# result_list = []
#
# for i in range(n):
#     result_list.append(simple_list)
#
# print(result_list)




# j = 0
# students = [["Jane", "B"], ["Kate", "B"], ["Alex", "C"], ["Elsa", "A"], ["Max", "B"], ["Chris", "A"]]
# students_list = []
# for i in students:
#     if i[1] == "A":
#         j += 1
#         students_list.append(i[0])
# print(students_list)
# # print([i[0] for i in students if i[1] == "A"])


# str1 = input()
# str2 = input()
# str3 = input()

# str1 = "input_1"
# str2 = "input_2"
# str3 = "input_3"

# lst = []
# lst1 = []
# lst2 = []
# lst3 = []
# lst.insert(0, str1)
# lst1.insert(0, str2)
# lst.insert(1, lst1)
# lst2.insert(0, str3)
# lst3.insert(0, lst2)
# lst.insert(2, lst3)
# print (lst)



# M = [[34, 77, 8,  45],
#      [10, 15, 93, 57],
#      [78, 82, 19, 98]]
#
# print(M[2][0])


# # Изменяемость списков и Неизменяемость кортежей
# f = [1, 2, 3, "A"]
# c = tuple(f)
#
# f[0] = 10
#
# print(c)
# print(f)
#
# # zip() - несколько итерируемых объектов
#
# d = ["a", "b"]
#     # zip(f, c, d, strict=True)
#     # strict=True - ошибка, если количество ит. элементов не одинакова
# for i, j, k in zip(f, c, d):
#     print(i, j, k)

# # enumerate()
#
# student_list = [1, 2, 3, "A"]
# for num, name in enumerate(student_list, start=1):
#     print(num, name)

# Vectors SUM

# v1 = (1, 5)
# v2 = (2, 3)
#
# v_sum = []
#
# for x, y in zip(v1, v2):
#     v_sum.append(x + y)
#     print(x + y)
# print(tuple(v_sum))

# str1 = input()
# str2 = input()
#
# str_name = []
#
# for i in zip(str1, str2):
#     str_name += i
# print("".join(str_name))

# # СЛОВАРИ
# # ----------------------------------------------------------------
#
# # Вставить в словарь
# alphabet = {}
# alphabet['alpha'] = 1
# alphabet['beta'] = 2
#
# print(alphabet)
# # Python 3.8 output: {'alpha': 1, 'beta': 2}
#
#
# # Перезаписать!!!!!
# alphabet['alpha'] = 'NEW YEAR'
# print(alphabet)
# # Python 3.8 output: {'alpha': 'NEW YEAR', 'beta': 2}
#
# my_pets = {'dog': {'name': 'Dolly', 'breed': 'collie'},
#            'cat': {'name': 'Fluffy', 'breed': 'maine coon'}}
#
# # Добавляем пару в dog -> count: '4'
# my_pets['dog']['count'] = 4
# print('my_pets:', my_pets)
#
# # Увеличение числового значения ключа
# my_pets['dog']['count'] += 6
# print('my_pets:', my_pets)
#
# # Добавление строки к строковому значению
# my_pets['dog']['name'] += '6'
# print('my_pets:', my_pets)
#
# # Значения в виде списка - нескольких значений
# my_dict = {'key1': ['value1', 'value2']}
#
# # Добавляем новой значение в список значений ключа -> 'new_value'
# my_dict['key1'].append(9)
# print('my_dict:', my_dict)
#
# # Увеличение числового значения ключа в списке
# my_dict['key1'][2] += 1
# print('my_dict:', my_dict)
#
# # Добавление строки к строковому значению в списке
# my_dict['key1'][1] += '_1'
# print('my_dict:', my_dict)

# import requests

# req = requests.get('https://seobells.com')
# print("url:", req.url)
# print("request:", req.request)
# print("object:", req)
# print("status_code:", req.status_code)
# # print("html:", req.text)
# print("encoding:", req.encoding)
# print("headers:", req.headers)
# print("connection:", req.connection)
# print("cookies:", req.cookies)
# print("history:", req.history)
#
# # url: https://seobells.com/
# # request: <PreparedRequest [GET]>
# # object: <Response [200]>
# # status_code: 200
# # encoding: UTF-8
# # headers: {'keep-alive': 'timeout=5, max=100', 'cache-control': 'public, max-age=0', 'expires': 'Fri, 25 Oct 2024 11:50:11 GMT', 'content-type': 'text/html; charset=UTF-8', 'last-modified': 'Fri, 04 Oct 2024 14:01:20 GMT', 'accept-ranges': 'bytes', 'content-encoding': 'gzip', 'vary': 'Accept-Encoding,Accept-Encoding', 'content-length': '60424', 'date': 'Fri, 25 Oct 2024 11:50:11 GMT', 'server': 'LiteSpeed', 'content-security-policy': 'upgrade-insecure-requests', 'x-turbo-charged-by': 'LiteSpeed'}
# # connection: <requests.adapters.HTTPAdapter object at 0x1038558e0>
# # cookies: <RequestsCookieJar[]>
# # history: []


# req = requests.get('https://mtlawoffice.com/')
# print("url:", req.url)
# print("request:", req.request)
# print("object:", req)
# print("status_code:", req.status_code)
# # print("html:", req.text)
# print("encoding:", req.encoding)
# print("headers:", req.headers)
# print("connection:", req.connection)
# print("cookies:", req.cookies)
# print("history:", req.history)
#
# # url: https://mtlawoffice.com/
# # request: <PreparedRequest [GET]>
# # object: <Response [200]>
# # status_code: 200
# # encoding: UTF-8
# # headers: {'Date': 'Fri, 25 Oct 2024 11:53:20 GMT', 'Content-Type': 'text/html; charset=UTF-8', 'Transfer-Encoding': 'chunked', 'Connection': 'keep-alive', 'Set-Cookie': 'AWSALBTG=kju0537q/XHogoPhvGSwEgBiakNNsk3vKhvGL6LM7jZ6CLZSK3X1K309S5yw++bh90hZJdGEpnHNwbNiT6ubfCu0h0r4y92+TL63kSFXcerXn0j93SZEWpgiqkXsYueNmX7hUmb8kne20TJx7FQ0xmIDJer9XRR1vSRBPEcFjTG2K9AF71o=; Expires=Fri, 01 Nov 2024 11:53:20 GMT; Path=/, AWSALBTGCORS=kju0537q/XHogoPhvGSwEgBiakNNsk3vKhvGL6LM7jZ6CLZSK3X1K309S5yw++bh90hZJdGEpnHNwbNiT6ubfCu0h0r4y92+TL63kSFXcerXn0j93SZEWpgiqkXsYueNmX7hUmb8kne20TJx7FQ0xmIDJer9XRR1vSRBPEcFjTG2K9AF71o=; Expires=Fri, 01 Nov 2024 11:53:20 GMT; Path=/; SameSite=None; Secure, AWSALB=OMDfc1U0MSLIPPrYb85n11JLzFEIg0RMzLFvn62tlo665pvf/E4K+OqLQdOUiULpDjEgKQt56J7fyal1fmGT04ZEzV1GWS+h7crIxkoWKRz+KXkJu+RFjbd3RKu7; Expires=Fri, 01 Nov 2024 11:53:20 GMT; Path=/, AWSALBCORS=OMDfc1U0MSLIPPrYb85n11JLzFEIg0RMzLFvn62tlo665pvf/E4K+OqLQdOUiULpDjEgKQt56J7fyal1fmGT04ZEzV1GWS+h7crIxkoWKRz+KXkJu+RFjbd3RKu7; Expires=Fri, 01 Nov 2024 11:53:20 GMT; Path=/; SameSite=None; Secure, PHPSESSID=u2ojfkorroq4umu12j5rbvogjk; path=/; secure; HttpOnly', 'strict-transport-security': 'max-age=63072000; includeSubdomains;', 'x-frame-options': 'SAMEORIGIN', 'expires': 'Fri, 25 Oct 2024 12:05:20 UTC', 'Cache-Control': 'max-age=1800', 'pragma': 'no-cache', 'octane-token': 'e4a99b4d74367c68160b5d55270aea9c2aada5398db03f3d1e9e84a898e93d81', 'octane': 'true', 'dynamix-cache': 'HIT', 'octane-version': '3.4', 'vary': 'Accept-Encoding', 'access-control-allow-origin': 'https://octaneforms.com', 'x-xss-protection': '1; mode=block', 'x-content-type-options': 'nosniff', 'referrer-policy': 'no-referrer-when-downgrade', 'cf-cache-status': 'DYNAMIC', 'Report-To': '{"endpoints":[{"url":"https:\\/\\/a.nel.cloudflare.com\\/report\\/v4?s=9yuLbN8xhZqgnYOzvfzQ9Pdf7BVYIxtljr7cBFP9R%2BTMdQGJYCEa6Fh4RVDhuNG37eS7zQZgQOUEyqRJ3KNCbbDHVCEFYXY3p1xLIxaUPqmrksBeqoo5mjAqMdeVRbZTlA%3D%3D"}],"group":"cf-nel","max_age":604800}', 'NEL': '{"success_fraction":0,"report_to":"cf-nel","max_age":604800}', 'Server': 'cloudflare', 'CF-RAY': '8d82016d4b6ebf21-WAW', 'Content-Encoding': 'gzip'}
# # connection: <requests.adapters.HTTPAdapter object at 0x1014cbbc0>
# # cookies: <RequestsCookieJar[<Cookie AWSALBTG=kju0537q/XHogoPhvGSwEgBiakNNsk3vKhvGL6LM7jZ6CLZSK3X1K309S5yw++bh90hZJdGEpnHNwbNiT6ubfCu0h0r4y92+TL63kSFXcerXn0j93SZEWpgiqkXsYueNmX7hUmb8kne20TJx7FQ0xmIDJer9XRR1vSRBPEcFjTG2K9AF71o= for mtlawoffice.com/>, <Cookie AWSALBTGCORS=kju0537q/XHogoPhvGSwEgBiakNNsk3vKhvGL6LM7jZ6CLZSK3X1K309S5yw++bh90hZJdGEpnHNwbNiT6ubfCu0h0r4y92+TL63kSFXcerXn0j93SZEWpgiqkXsYueNmX7hUmb8kne20TJx7FQ0xmIDJer9XRR1vSRBPEcFjTG2K9AF71o= for mtlawoffice.com/>, <Cookie AWSALB=OMDfc1U0MSLIPPrYb85n11JLzFEIg0RMzLFvn62tlo665pvf/E4K+OqLQdOUiULpDjEgKQt56J7fyal1fmGT04ZEzV1GWS+h7crIxkoWKRz+KXkJu+RFjbd3RKu7 for mtlawoffice.com/>, <Cookie AWSALBCORS=OMDfc1U0MSLIPPrYb85n11JLzFEIg0RMzLFvn62tlo665pvf/E4K+OqLQdOUiULpDjEgKQt56J7fyal1fmGT04ZEzV1GWS+h7crIxkoWKRz+KXkJu+RFjbd3RKu7 for mtlawoffice.com/>, <Cookie PHPSESSID=u2ojfkorroq4umu12j5rbvogjk for mtlawoffice.com/>]>
# # history: []

# # Type casting
# n = 2.777
# print(str(float(int(n))))
# # a = "10.0"
# # print(int(a)) # Error
#
# a = "0"
# print(int(a))
# a = "215"
# print(int(a))
# a = "-1"
# print(int(a))
# a = "15.5"
# print(float(a))
#
#
# word = input()
#
# # Change the next line
# print(word * 2)

# NumPy
#
# import numpy as np
# from matplotlib.lines import drawStyles

#
# first = np.array([1, 2, 3, 4, 5])
# second = np.array([[1, 1, 1],
#                    [2, 2, 2]])
#
# print(first.shape)  # (5,)
# print(second.shape)  # (2, 3)
#
#
# thrid = np.array([[1, 1, 1],
#                   [2, 2, 2],
#                   [2, 2, 2]
#                   ])
#
# print(thrid.shape)  # (3, 3)
#
#
# print(first.shape[0], 'elements')  # 5 elements
# print(second.shape[1], 'column')  # 3 column
# print(thrid.shape[0], 'row')  # 3 row
#
#
# print(first.ndim, 'dimension')  # 1
# print(second.ndim, 'dimensions')  # 2
# print(thrid.ndim, 'dimensions')  # 2!!!!!!!!!!!!!!!!!
#
#
# print(len(first), first.size, '')  # 5 5
# print(len(second), second.size, '')  # 2 6
# print(len(thrid), thrid.size, '')  # 3 9
#

# middle level
#
# arr = np.zeros((4, 3, 2), dtype=int)  # 3D array with zeros of the specified shape
#
# arr_new = np.array((4, 3, 2), dtype=int)  # 3D array with zeros of the specified shape
# print(arr_new)
#
# rnd_arr = np.random.randint(
#     0, 20, (4, 3, 2)
# )  # 3D array with random integers from 0 to 20
#
# print(arr)
# print(arr.ndim, 'dimensions')  # 3!!!!!!!!!!!!!!!!!
#
# ###
#
# a = np.array([9, 99, 999])
# print(type(a))
# print(a.itemsize, a.size, a.itemsize * a.size) # 24bite - array size
#
# ###
#
# n = np.zeros((3, 2, 6, 1))
#
# print(n.ndim, n.size, n.shape[0], n.ndim + n.size + n.shape[0])
#
#
# ####
#
# # Input1 :
# Input1_1 = [1, 2, 3]
# Input1_2 = [4, 5, 6]
# # Expected Output: 'Both arguments are lists, not arrays'
#
# # Input:
# Input2_1 = np.array([1, 2, 3])
# Input2_2 = [4, 5, 6]
# # Expected Output: 'One argument is a list'
#
# # Input:
# Input3_1 = np.array([1, 2, 3])
# Input3_2 = np.array([4, 5, 6])
# # Expected Output: array([5, 7, 9])
#
# one_list_warning = "One argument is a list"
# two_lists_warning = "Both arguments are lists, not arrays"
#
# def custom_sum(arg1, arg2):
#     if type(arg1) == np.ndarray and type(arg2) == np.ndarray:
#         return arg1 + arg2
#     elif type(arg1) == np.ndarray and type(arg2) == list or type(arg1) == list and type(arg2) == np.ndarray:
#             return one_list_warning
#     elif type(arg1) == list and type(arg2) == list:
#             return two_lists_warning
#
#
# print(custom_sum(Input1_1, Input1_2))
# print(custom_sum(Input2_1, Input2_2))
# print(custom_sum(Input3_1, Input3_2))
#
# #
# print(type(Input1_1), type(Input1_2), type(Input2_1), type(Input2_2), type(Input3_1), type(Input3_2))
# print(Input1_1 + Input1_2)
# print(type(Input1_1) == list)
# # print(isinstance(Input3_2, type))
#
# array_1 = np.arange(5)
# print(array_1)  # [0 1 2 3 4]
#
# array_4 = np.linspace(21, 23, num=5)
# print(array_4)   # [21.  21.5 22.  22.5 23. ]
#
#
# array_6 = np.ones((3, 2))
# print(array_6)
# array_66 = np.ones((3, 2, 6))
# print(array_66, 'END')
#
# array_7 = np.zeros(7)
# print(array_6)
# # [[1. 1.]
# #  [1. 1.]
# #  [1. 1.]]
# print(array_7)  # [0. 0. 0. 0. 0. 0. 0.]
#
#
# x = np.array([[1, 1, 1], [2, 2, 2]])
# y = np.array([1, 2, 3, 4, 5])
# array_8 = np.ones_like(x)
# array_9 = np.zeros_like(y)
# print(array_8)
# # [[1 1 1]
# #  [1 1 1]]
# print(array_9)  # [0 0 0 0 0]
#
# array_10 = np.array([[1, 2], [3, 4]])
# lst1 = array_10.tolist()
# print(lst1)  # [[1, 2], [3, 4]]
# print(type(lst1[0]))  # list
#
#
# array_11 = np.array([[5, 2], [7, 4]])
# lst2 = list(array_11)
# print(lst2)  # [array([5, 2]), array([7, 4])]
# print(type(lst2[0]))  # <class 'numpy.ndarray'>!!!!!
#
#
#
# array_12 = np.array([[1, 12, 31], [4, 45, 64], [0, 7, 89]])
# print(array_12[2, 2])  # 89
# print(array_12[2][2])  # 89
#
#
#
# array_13 = np.array([[[1, 12, 31], [4, 45, 64], [0, 7, 89]]])
# print(array_13[0, 1, 1])  # 45
# print(array_13[0][1][1])  # 45
#
#
# array_14 = np.array([[100, 101, 102],
#                  [103, 104, 105],
#                  [106, 107, 108],
#                  [109, 110, 111]])
# print(array_14[1:3, 1])   # [104 107]
#
#
# array_15 = np.array([[[1, 2, 3, 4, 5],
#                  [6, 7, 8, 9, 10]],
#                  [[11, 12, 13, 14, 15],
#                  [16, 17, 18, 19, 20]]])
# print(array_15[-1, :, 1:4])
# # [[12 13 14]
# #  [17 18 19]]
#
#
# # two-dimensional array
# array_16 = np.array([[1, 2, 3, 4, 5],
#                  [5, 4, 3, 2, 1],
#                  [6, 7, 8, 9, 10],
#                  [10, 9, 8, 7, 6],
#                  [11, 12, 13, 14, 15]])
# print(array_16[::2, ::2])
# # [[ 1  3  5]
# #  [ 6  8 10]
# #  [11 13 15]]




#
# n = int(input())
# bit = int(input())
#
# arr = np.full((n,n), bit, dtype=float)
# print(arr)
#



#
# n = int(input())
# bit = int(input())
# a = np.array([[1, 3, 4],
#               [45, 66, 76],
#               [0, 9, 4],
#               [12, 14, 90],
#               [39, 71, 83],
#               [27, 20, 5]])
# print(a[n, bit])
#


#
# a = np.array([[[10, 11, 12], [13, 14, 15], [16, 17, 18]],
#               [[20, 21, 22], [23, 24, 25], [26, 27, 28]],
#               [[30, 31, 32], [33, 34, 35], [36, 37, 38]],
#               [[40, 41, 42], [43, 44, 45], [46, 47, 48]],
#               [[50, 51, 52], [53, 54, 55], [56, 57, 58]],
#               [[60, 61, 62], [63, 64, 65], [66, 67, 68]],
#               [[70, 71, 62], [73, 74, 65], [76, 77, 78]],
#               [[80, 81, 62], [83, 84, 85], [86, 87, 88]]])
#
# M = int(input())
# s_row = int(input())
# el_num = int(input())
#
# print(a[::M, ::s_row, el_num])



# strat_el = int(input())
# end_el = int(input())
# step = int(input())
#
# arr = np.linspace(strat_el, end_el, step, dtype=float)
#
# print(arr[-2])

#
# arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
# if arr2d.shape == (4,3):
#     print(arr2d[:2, 1:])
# else:
#     print(arr2d[-2:, :2])

#
# import numpy
# # from numpy.linalg import inv
#
# A = numpy.array([[1, 2], [2, 3]])
# AI = inv(A)
# tr = numpy.trace(AI)
# print(tr)



# a1 = int(input())
# a2 = int(input())
# a3 = int(input())
# a4 = int(input())
# a5 = int(input())
# a6 = int(input())
#
# M1 = np.array([[a1, a2], [a3, a4]])
# M2v = np.array([a5, a6])
# det = M1 / M2v
#
# print(det.T)


# a1 = int(input())
# a2 = int(input())
# a3 = int(input())
#
# M = np.array([a1, a2, a3])
# print(M.std())


# array = np.array([[[14, 5], [8, 0]], [[13, 7], [18, 5]]])
# print(array.sum(axis=0))
# print(array.sum(axis=1))
# print(array.sum(axis=2))


# axis = int(input())
# array = np.array([[[7, 8], [13, 9]], [[9, 5], [15, 14]]])
# print(array.mean(axis=axis))


# a1 = int(input())
# a2 = int(input())
# a3 = int(input())
#
# M = np.array([a1, a2, a3])
# print(M.max())
# print(M.argmax())


# axis = int(input())
# array = np.array([[[13, 14], [34, 35]], [[9, 9], [5, 0]]])
# print(array.sum(axis=axis))

#
# name = ['M', 'A', 'R', 'C', 'O']
# # modify the next line
# print(*name, sep="-",  end="!")



# numbers = [int(x) for x in input().split()]
# # print all numbers without spaces
# print(*numbers, sep="")


# line1 = "Night, square, apothecary, lantern,"
# line2 = "Its meaningless and pallid light."
# line3 = "Return a half a lifetime after – "
# line4 = "All will remain. A scapeless rite."
#
# # your one print() statement here
# print(line1, line2, line3, line4, sep="\n")


# print('I am the', 'test', sep='!')


# rent = input()
# spaces = int(input())
# count = ' ' * spaces
# print(*rent, sep=count)

############################################################################
# ######## pandas
############################################################################
#
# import pandas as pd
# import polars as pl
#
# # EXCEL
# df = pd.read_excel('/Users/palpme/gsc_data_full.xlsx')
# # print(df)
#
# print(df.head())
# print('START INFO')
# df.info()
# print('END INFO')
# print(df.describe())
#
#
# # # CSV
# # df = pd.read_csv('file_path.csv')
# #
# # # JSON
# # df = pd.read_json('file_path.json')
#
# # # import pandas as pd
# # from sqlalchemy import create_engine
# #
# # engine = create_engine('database_connection_string')
# # query = "SELECT * FROM table_name"
# # df = pd.read_sql(query, engine)
#
# # df_cars = pd.read_csv('./cars.csv', sep=',') # чтение csv файла (может быть ';')
# # df_cars = pd.read_csv('./cars.csv', delimiter=';')
#
#
# # # POLARS !!!!!!!!!!!!!
# #
# # df = pl.read_excel('/Users/palpme/gsc_data_full.xlsx')
# # print(df)

#
# df = pd.read_csv('/Users/palpme/hyperskill-dataset-105528291.txt', index_col='Name' )
# print(df.head(10))

# df = pd.read_csv('/Users/palpme/hyperskill-dataset-105528862.txt')
# print(df.head(10))
# df.info()
# print('END INFO')
# print(df.describe())

# df = pd.read_excel('/Users/palpme/gsc_data_full.xlsx')
# print(df)
# df['name'] = df['name'].astype('category')
# print(df.dtypes)
# print(df.memory_usage(deep=True).sum())
#
# df['name'] = df['name'].astype('category')
# print(df.dtypes)
# print(df.memory_usage(deep=True).sum())
#
# #
# # # Converting the 'Date' column to datetime
# df['date'] = pd.to_datetime(df['date'])
#
# # Formatting the 'Date' column
# df['date'] = df['date'].dt.strftime('%B %d, %Y')
#
# print(df)
# print(df.memory_usage(deep=True).sum())
# print(df.dtypes)

# print(df.isnull())

# # Remove zeros
# df = df.dropna()
# print(df)
#
# # Fill null values before converting
# df['A'] = df['A'].fillna(0).astype(int)

# Forward fill (using the previous non-null value)
# df['Salary'] = df['Salary'].fillna(method='ffill')


# EXCEPTION

# number_one = int(input("Please, enter the first number: "))
# number_two = int(input("Please, enter the second number: "))
# try:
#     result = number_one / number_two
# except Exception as e:
#         print(f"Error during impersonation: {e}")
# else:
#     print("The result of your division is: ", result)
# finally:
#     print("It is done through finally ***Thanks for using our calculator! Come again!")


# try:
#     name, surname = input().split()
# except ValueError:
#     print(f"You need to enter exactly 2 words. Try again!")
# else:
#     print("Welcome to our party", name, surname)


try:
    exception_test()
except Exception:
    print("Exception")
except BaseException:
    print("BaseException")