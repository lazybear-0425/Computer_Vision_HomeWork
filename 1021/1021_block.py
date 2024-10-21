import cv2
import numpy as np


img = cv2.imread('1021/example/block.jpg')
h = img.shape[0]
w = img.shape[1]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# thres = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 2)
_, thres = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thres, connectivity=8, ltype=None)

print(max(stats[:, 4])) # x、y、width、height和面积

# 參考自
# https://gitcode.csdn.net/66c9b67013e4054e7e7d59c2.html?dp_token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpZCI6MzE3NzA3NCwiZXhwIjoxNzMwMTAyODA5LCJpYXQiOjE3Mjk0OTgwMDksInVzZXJuYW1lIjoiMjMwMV83OTAyODYwOSJ9.yk7OgQeVEITBqDDS5W3mFSkjIFLcHnew5SVTZ5647zQ&spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7Ebaidujs_baidulandingword%7Eactivity-3-127225934-blog-106023288.235%5Ev43%5Epc_blog_bottom_relevance_base1&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7Ebaidujs_baidulandingword%7Eactivity-3-127225934-blog-106023288.235%5Ev43%5Epc_blog_bottom_relevance_base1&utm_relevant_index=6
output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
want_group = []
for i in range(1, num_labels):
    if stats[i][4] > 80000: continue
    if stats[i][4] < 3000: continue

    want_group.append(i)
    mask = labels == i
    output[:, :, 0][mask] = np.random.randint(0, 255)
    output[:, :, 1][mask] = np.random.randint(0, 255)
    output[:, :, 2][mask] = np.random.randint(0, 255)
print('OK') # debug
cv2.imwrite('1021/result/find_connect.jpg', output)

group = {}
for i in range(h):
    for j in range(w):
        group_num = labels[i][j]
        if(group_num == 0): continue
        if(group_num not in want_group): continue
        if group_num not in group:
            group[group_num] = []
        group[group_num].append([j, i])
print('OK') # debug

group_dis = {}
for g in group:
    nodes = group[g]
    max_dis = 0
    max_node = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            node_1 = nodes[i]
            node_2 = nodes[j]
            dis = (node_1[0] - node_2[0]) ** 2 + (node_1[1] - node_2[1]) ** 2
            if(max_dis < dis): 
                max_dis = dis
                max_node = [node_1, node_2]
    group_dis[g] = max_node
print('OK') # debug

for g in group:
    nodes = group[g]
    node_1 = group_dis[g][0]
    node_2 = group_dis[g][1]
    max_dis = 0; max_node = []
    min_dis = 10000000; min_node = []
    
    abc = np.cross(np.array([node_1[0], node_1[1], 1]),
                   np.array([node_2[0], node_2[1], 1]))
    abc = abc / (abc[0] ** 2 + abc[1] ** 2) ** 0.5
    def func(x, y):
        return abc[0] * x + abc[1] * y + abc[2]
    for node in nodes:
        dis = func(node[0], node[1])
        if max_dis < dis:
            max_dis = dis
            max_node = node
        if min_dis > dis:
            min_dis = dis
            min_node = node
    group_dis[g].append(min_node)
    group_dis[g].append(max_node)
print('OK') # debug

find_coner = img.copy()
for g in group_dis:
    for i in range(4):
        node = group_dis[g][i]
        cv2.circle(find_coner, (node[0], node[1]), 13, (0, 0, 255), -1)
cv2.imwrite('1021/result/find_coner.jpg', find_coner)

def convert(pos):    
    h = 48
    w = 48
    origin_cord = np.array([[0, 0], [0, h], [w, 0], [w, h]], dtype=np.float32)
    my_cord = np.array(pos, dtype=np.float32)

    M = cv2.getPerspectiveTransform(my_cord, origin_cord)
    # M2 = cv2.findHomography(my_cord, origin_cord)

    persp = cv2.warpPerspective(thres, M, (w, h))

    one_or_zero = np.zeros((6, 6))
    for i in range(w):
        for j in range(h):
            if persp[i][j] == 255: # 因為之前有黑白顛倒
                one_or_zero[(i // 8)][(j // 8)] += 1
    one_or_zero = one_or_zero / 64

    num = 0
    for i in range(1, 5):
        for j in range(1, 5):
            num *= 2
            if one_or_zero[i][j] >= 0.5: num += 1
            else: num += 0
    return num, f'{num:b}'
    # print(num)
    # print(f'{num:b}')

for g in group_dis:
    group_dis[g].sort()
    node_1 = group_dis[g][0]
    node_2 = group_dis[g][1]
    num, binary = convert(group_dis[g])
    cv2.putText(img, f'{num}', (int((node_1[0] + node_2[0]) / 2), node_1[1]), cv2.FONT_HERSHEY_TRIPLEX, 1, 
                (0, 0, 255), 3)
    print(binary)

cv2.imwrite('1021/result/decode.jpg', img)

# contours, hierarchy = cv2.findContours(thres, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# contour = cv2.drawContours(img, contours, -1, (0, 0, 255), 5)

# 目標

# convex -> 減少點的數量(-> 快)