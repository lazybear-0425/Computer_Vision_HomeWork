# 備註 : 這邊是整理成function版本，比較容易讀
# 可以從 if __name__ == '__main__': 看
import cv2
import numpy as np

def select_wanted(num_labels, labels, stats, img):
    # 參考自
    # https://pse.is/6lp6v8
    output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    want_group = []
    for i in range(1, num_labels):
        # 篩選掉面積太大或太小，stats[i][4] -> 面積
        if stats[i][4] > 11000: continue
        if stats[i][4] < 3000: continue

        want_group.append(i)
        mask = labels == i
        output[:, :, 0][mask] = np.random.randint(0, 255)
        output[:, :, 1][mask] = np.random.randint(0, 255)
        output[:, :, 2][mask] = np.random.randint(0, 255)
    cv2.imwrite('1021/result/find_connect.jpg', output)
    print('Save \033[33mfind_connect.jpg\033[0m') # debug
    return want_group

def select_contour(labels, want_group, h, w):
    group = {}
    for i in range(h):
        for j in range(w):
            group_num = labels[i][j]
            if(group_num == 0): continue
            if(group_num not in want_group): continue
            if group_num not in group: # python要手動新增好討厭QQ
                group[group_num] = []
            if i - 1 >= 0 and i + 1 < h and j - 1 >= 0 and j + 1 < w: # 留下輪廓 => 加速!
                if labels[i][j] == labels[i - 1][j] == labels[i][j - 1] == labels[i + 1][j] == labels[i][j + 1]: continue
            group[group_num].append([j, i])
    return group

def print_contour(group, h, w):
    contour = np.zeros((h, w, 3), dtype=np.uint8)
    for g in group:
        for node in group[g]:
            i = node[0]
            j = node[1]
            contour[j, i, :] = 255
    contour = cv2.dilate(contour, (5, 5)) # 因為原圖太糊了QQ
    cv2.imwrite('1021/result/contour.jpg', contour)
    print('Save \033[33mcontour.jpg\033[0m') # debug

def select_coner(group):
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
    return group_dis

def print_coner(group_dis):
    find_coner = img.copy()
    for g in group_dis:
        for i in range(4):
            node = group_dis[g][i]
            cv2.circle(find_coner, (node[0], node[1]), 10, (0, 0, 255), -1)
    cv2.imwrite('1021/result/find_coner.jpg', find_coner)
    print('Save \033[33mfind_coner.jpg\033[0m') # debug
    
def erode_second(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    erode = cv2.erode(img, kernel)
    return erode

def convert(pos, g):    
    h = 48
    w = 48
    origin_cord = np.array([[0, 0], [0, h], [w, 0], [w, h]], dtype=np.float32)
    my_cord = np.array(pos, dtype=np.float32)

    # M = cv2.getPerspectiveTransform(my_cord, origin_cord)
    M, _ = cv2.findHomography(my_cord, origin_cord)

    persp = cv2.warpPerspective(thres, M, (w, h))

    if my_cord[0][0] >= 1120 and my_cord[0][1] >= 790:
        persp = erode_second(persp)

    one_or_zero = np.zeros((6, 6))
    for i in range(w):
        for j in range(h):
            if persp[i][j] == 0: # 因為之前有黑白顛倒
                one_or_zero[(i // 8)][(j // 8)] += 1
    one_or_zero = one_or_zero / 64

    num = 0
    binary = ''
    for i in range(1, 5):
        for j in range(1, 5):
            num *= 2
            if one_or_zero[i][j] >= 0.5: num += 1; binary += '1'
            else: num += 0; binary += '0'
    cv2.imwrite(f'1021/result/blocks/{g}.jpg', cv2.bitwise_not(persp))
    return num, binary

def sort(l): # 自定義sort方法
    cord = np.array(l) # 弄成numpy好算很多XD
    max = np.argmax(np.sum(cord, axis = 1))     
    cord[3, :], cord[max, :] = cord[max, :].copy(), cord[3, :].copy()
    min = np.argmin(np.sum(cord, axis = 1))     
    cord[0, :], cord[min, :] = cord[min, :].copy(), cord[0, :].copy()
    if cord[1][1] < cord[2][1]:
        cord[1, :], cord[2, :] = cord[2, :].copy(), cord[1, :].copy()
    return cord.tolist()

def print_decode(group_dis, img):
    for g in group_dis:
        group_dis[g] = sort(group_dis[g])
        node_1 = group_dis[g][0]
        node_2 = group_dis[g][1]
        num, binary = convert(group_dis[g], g)
        cv2.putText(img, f'{num}', (int((node_1[0] + node_2[0]) / 2), node_1[1] - 20), cv2.FONT_HERSHEY_TRIPLEX, 1.5, 
                    (0, 0, 255), 2)
        print(f'{num:016b} => {num:8d}')
    print('Save \033[33mblocks/g.jpg\033[0m') # debug
    cv2.imwrite('1021/result/decode.jpg', img)
    print('Save \033[33mdecode.jpg\033[0m') # debug

# contours, hierarchy = cv2.findContours(thres, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# contour = cv2.drawContours(img, contours, -1, (0, 0, 255), 5)

# convex -> 減少點的數量(-> 快)

if __name__ == '__main__':
    img = cv2.imread('1021/example/block.jpg')
    h = img.shape[0]
    w = img.shape[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 也能用adaptive，blocksize要調大
    # thres = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 301, 2)
    _, thres = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thres = cv2.erode(thres, kernel)

    cv2.imwrite('1021/result/threshold.jpg', thres)
    print('Save \033[33mthreshold.jpg\033[0m') # debug

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thres, connectivity=8, ltype=None)
    print(f'number of labels = \033[35m{num_labels}\033[0m') # x、y、width、height和面積

    want_group = select_wanted(num_labels, labels, stats, img)
    group = select_contour(labels, want_group, h, w)
    print_contour(group, h, w)
    group_dis = select_coner(group)
    print_coner(group_dis)
    print_decode(group_dis, img)