import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
from random import randint
import numpy as np
import sys
import mediapipe as mp
# import tkinter as tk # 取消用tkinter
import math
import pygame

# ==== the author of 3D code ============================================================
# Function for stereo vision and depth estimation
sys.path.append('term_project/')
import triangulation as tri
import calibration # 用於校正用(?)這部分不確定要不要用

# Stereo vision setup parameters
B = 8               #Distance between the cameras [cm]
f = 4              #Camera lense's focal length [mm]
alpha = 56.6        #Camera field of view in the horisontal plane [degrees]
# ==== the author of 3D code ============================================================

def cal_angle(v1, v2):
    v1_x = v1[0]
    v1_y = v1[1]
    v2_x = v2[0]
    v2_y = v2[1]
    try: # v1·v2 = ||v1|| * ||v2|| * cos(theta) # math.degree: 弧度轉角度
        angle = math.degrees(math.acos((v1_x * v2_x + v1_y * v2_y) / 
                                        ((v1_x ** 2 + v1_y ** 2) ** 0.5 * (v2_x ** 2 + v2_y ** 2) ** 0.5)))
    except:
        angle = 180
    return angle

def pygame_word(surface, string='', color=(100, 0, 255), size=60, pos_x=None, pos_y=None):
    basicFont = pygame.font.SysFont("Arial", size)
    text = basicFont.render(string, True, color)
    textRect = text.get_rect()
    if pos_x: textRect.centerx = pos_x
    else: textRect.centerx = surface.get_rect().centerx #screen為初始化時視窗
    if pos_y: textRect.centery = pos_y
    else: textRect.centery = surface.get_rect().centery
    surface.blit(text, textRect)

# def tkinter_word(root, text):
#     myLabel = tk.Label(root, text=text, font=('Arial',10,'bold'))
#     myLabel.pack(padx=10)
#     root.update()

def show_word(img, text):
    if not text: return
    cv2.putText(img, text, (img.shape[0] // 4, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

def detect_hand(img, hands):
    global my_draw
    hand_record = []
    result = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if result.multi_hand_landmarks:
        for hand_landmark in result.multi_hand_landmarks:
            for hand in hand_landmark.landmark:
                hand_record.append([hand.x, hand.y])
            my_draw.draw_landmarks(img, hand_landmark, mp_hand.HAND_CONNECTIONS, 
                                            landmark_drawing_spec=my_draw_style.get_default_pose_landmarks_style())
        return hand_record
    return []

def cal_finger_angle(hand_record):
    if hand_record:
        finger = []
        finger.append(cal_angle((hand_record[0][0] - hand_record[2][0], hand_record[0][1] - hand_record[2][1]), (hand_record[3][0] - hand_record[4][0], hand_record[3][1] - hand_record[4][1])))
        finger.append(cal_angle((hand_record[0][0] - hand_record[6][0], hand_record[0][1] - hand_record[6][1]), (hand_record[7][0] - hand_record[8][0], hand_record[7][1] - hand_record[8][1])))
        finger.append(cal_angle((hand_record[0][0] - hand_record[9][0], hand_record[0][1] - hand_record[9][1]), (hand_record[11][0] - hand_record[12][0], hand_record[11][1] - hand_record[12][1])))
        finger.append(cal_angle((hand_record[0][0] - hand_record[13][0], hand_record[0][1] - hand_record[13][1]), (hand_record[15][0] - hand_record[16][0], hand_record[15][1] - hand_record[16][1])))
        finger.append(cal_angle((hand_record[0][0] - hand_record[17][0], hand_record[0][1] - hand_record[17][1]), (hand_record[19][0] - hand_record[20][0], hand_record[19][1] - hand_record[20][1])))    
        return finger
    return []

def reset_record(record):
    record = {'ok':0, 'zero':0, 'one':0, 'two':0, 'three':0, 'four':0, 'five':0}

def cal_finger(finger, state, record, threshold):
    if state == 'ok' and (finger[2] < thres_angle and finger[3] < thres_angle and finger[4] < thres_angle) and (finger[1] >= thres_angle and finger[0] >= thres_angle * 0.4):
        record['ok'] += 1
        if record['ok'] >= threshold: return True
    elif state == 'zero' and (finger[0] >= thres_angle and finger[1] >= thres_angle and finger[2] >= thres_angle and finger[3] >= thres_angle and finger[4] >= thres_angle):
        reset_record(record)
    elif state == 'one' and (finger[1] < thres_angle) and (finger[0] >= thres_angle and finger[2] >= thres_angle and finger[3] >= thres_angle and finger[4] >= thres_angle):
        record['one'] += 1
        if record['one'] >= threshold: return True
    elif state == 'two' and (finger[3] < thres_angle and finger[2] < thres_angle) and (finger[0] >= thres_angle and finger[1] >= thres_angle and finger[4] >= thres_angle):
        record['two'] += 1
        if record['two'] >= threshold: return True
    elif state == 'two' and (finger[1] < thres_angle and finger[2] < thres_angle) and (finger[0] >= thres_angle and finger[3] >= thres_angle and finger[4] >= thres_angle):
        record['two'] += 1
        if record['two'] >= threshold: return True
    elif state == 'three' and (finger[1] < thres_angle and finger[2] < thres_angle and finger[3] < thres_angle) and (finger[0] >= thres_angle and finger[4] >= thres_angle):
        record['three'] += 1
        if record['three'] >= threshold: return True
    elif state == 'four' and (finger[1] < thres_angle and finger[2] < thres_angle and finger[3] < thres_angle and finger[4] < thres_angle) and (finger[0] >= thres_angle):
        record['four'] += 1
        if record['four'] >= threshold: return True
    elif state == 'five' and (finger[1] < thres_angle and finger[2] < thres_angle and finger[3] < thres_angle and finger[4] < thres_angle and finger[0] < thres_angle) and (True):
        record['five'] += 1
        if record['five'] >= threshold: return True
    return False

def direct_snack(snack_pos, x1, y1, game_level):
    step = 2 * game_level
    if x1 >= 0 and y1 >= 0: snack_pos = [snack_pos[0] + step, snack_pos[1] + step]
    elif x1 >= 0 and y1 < 0: snack_pos = [snack_pos[0] - step, snack_pos[1] + step]
    elif x1 < 0 and y1 >= 0: snack_pos = [snack_pos[0] + step, snack_pos[1] - step]
    else: snack_pos = [snack_pos[0] - step, snack_pos[1] - step]
    return snack_pos

def isExtrapolation(node_1, radius_1, node_2, radius_2): # 判斷是否外離
    o1o2 = ((node_1[0] - node_2[0]) ** 2 + (node_1[1] - node_2[1]) ** 2) ** 0.5
    return o1o2 > (radius_1 + radius_2)

def cal_boundBox(img, hand_record):                           # 記得這邊的shape要顛倒放
    real_hand_record = np.array(hand_record, dtype=np.float32) * np.array([img.shape[1], img.shape[0]]) 
    boundBox = cv2.boundingRect(real_hand_record.astype(np.float32)) # shape: (4, )
    [x, y, w, h] = boundBox
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3)
    return boundBox

def cal_3d_depth(img, img_other, hand_record, hand_record_other):
    boundBox = cal_boundBox(img, hand_record)
    center_point = (boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)
    boundBox_other = cal_boundBox(img_other, hand_record_other)
    center_point_other = (boundBox_other[0] + boundBox_other[2] / 2, boundBox_other[1] + boundBox_other[3] / 2)
    depth = tri.find_depth(center_point, center_point_other, img, img_other, B, f, alpha)
    # print(f'Depth: {round(depth,2)}')
    return round(depth, 2)
# ========= main ==================================================================
cap = cv2.VideoCapture(0)
cap_other = cv2.VideoCapture(1)
cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
my_draw = mp.solutions.drawing_utils    # 繪圖方式
my_draw_style = mp.solutions.drawing_styles # 繪圖樣式
mp_hand = mp.solutions.hands
mp_hand_other = mp.solutions.hands
# root = tk.Tk(); tkinter_word(root, 'message:\n請比"OK"表示開始遊戲')
start_game = 0
thres_angle = 50
game_level = 0
score = 0
snack_position = [0, 0]
fruit = []
minus_fruit = []
life = 0
counter = 0
minus_counter = 1
(x1, y1) = (randint(-1, 1), randint(-1, 1))
# score_strvar = tk.StringVar(); score_strvar.set(f'Score: {score}')
# depth_strvar = tk.StringVar(); depth_strvar.set(f'Depth: wait...')
record = {'ok':0, 'zero':0, 'one':0, 'two':0, 'three':0, 'four':0, 'five':0}

pygame.init()
pygame_screen = pygame.display.set_mode((cap_width, cap_height)); pygame_screen.fill((255, 255, 255))
minus_ball = pygame.image.load('term_project/images/minus_ball.png').convert_alpha()
minus_ball = pygame.transform.scale(minus_ball, (20, 20))
plus_ball = pygame.image.load('term_project/images/plus_ball.png').convert_alpha()
plus_ball = pygame.transform.scale(plus_ball, (20, 20))


if not cap.isOpened() or not cap_other.isOpened():
    print('攝影機開啟失敗')
    exit()
    
# 記住這邊要開兩個hand，如果設成max_num_hands=2會錯誤
with mp_hand.Hands(max_num_hands=1, min_detection_confidence=0.6) as hands:
    with mp_hand_other.Hands(max_num_hands=1, min_detection_confidence=0.6) as hands_other:
        while True:
            ret, img = cap.read()
            ret_other, img_other = cap_other.read()
            img = cv2.flip(img, 1)
            img_other = cv2.flip(img_other, 1)
            if not ret or not ret_other:
                print('error')
                exit()
            # 校正
            # img, img_other = calibration.undistortRectify(img, img_other)
            #======================= cal hand node =============================
            hand_record = detect_hand(img, hands) # 獲得21個點
            finger = cal_finger_angle(hand_record) # 計算手指彎曲程度
            hand_record_other = detect_hand(img_other, hands_other)
            #======================= cal 3D depth ==============================
            if hand_record and hand_record_other: # 去計算bound box -> 才能計算深度
                depth = cal_3d_depth(img, img_other, hand_record, hand_record_other)
                # depth_strvar.set(f'Depth: {depth}')
                pygame_word(pygame_screen, f'Depth: {abs(depth)}', (0, 0, 0), 22, 100, 30)
            else: 
                # depth_strvar.set(f'Depth: Loss Hand info')
                pygame_word(pygame_screen, 'Depth: Loss Hand info', (0, 0, 0), 22, 100, 30)
            # root.update()
            #======================= cal 3D depth ==============================
            if start_game == 0:
                pygame_word(pygame_screen, 'Indicate "OK" to start game', (0, 0, 0))
                
                # 因為OK的拇指不明顯
                if finger and cal_finger(finger, 'ok', record, 20):
                    start_game = 1
                    # tkinter_word(root, 'Choose Game Level! (1 - 5)')
                    reset_record(record)
            elif start_game == 1: # 判斷遊玩等級
                pygame_word(pygame_screen, 'Choose Game Level (1-5)', (0, 0, 0))
                if finger:
                    if cal_finger(finger, 'one', record, 20):
                        start_game = 2; game_level = 1
                    elif cal_finger(finger, 'two', record, 20):
                        start_game = 2; game_level = 2
                    elif cal_finger(finger, 'three', record, 20):
                        start_game = 2; game_level = 3
                    elif cal_finger(finger, 'four', record, 20):
                        start_game = 2; game_level = 4
                    elif cal_finger(finger, 'five', record, 20):
                        start_game = 2; game_level = 5
                    if game_level != 0:
                        # tkinter_word(root, f'You Choose Level {game_level}')
                        life = 6 - game_level
                        snack_position = [randint(img.shape[0] // 4, img.shape[0] // 4 * 3), randint(img.shape[1] // 4, img.shape[1] // 4 * 3)]
                        for i in range(10 - game_level):
                            fruit.append([randint(30, img.shape[0] - 30), randint(30, img.shape[1] - 30)])
                        # tk.Label(root, textvariable=score_strvar, font=('Arial',20,'bold')).pack()
                        for i in range(5 + game_level):
                            minus_fruit.append([randint(30, img.shape[0] - 30), randint(30, img.shape[1] - 30)])
            elif start_game >= 2 and start_game <= 4: # 倒數計時 3, 2, 1
                pygame.draw.circle(pygame_screen, (0, 255, 0), (snack_position[1], snack_position[0]), 14 - game_level, 0)
                pygame_word(pygame_screen, f'{5 - start_game}', (255, 100, 0), 132)
                pygame_word(pygame_screen, f'You Choose Level {game_level}', (255, 100, 0), size=40, pos_y=pygame_screen.get_rect().centery + 70)
                
                # cv2.putText(img, f'{5 - start_game}', (img.shape[1] // 2 - 80, img.shape[0] // 2 + 100), cv2.FONT_HERSHEY_COMPLEX,
                #             8, (0, 100, 255), 18)
                # cv2.circle(img, (snack_position[1], snack_position[0]), 14 - game_level, (0, 255, 0), -1)
                counter += 1
                if counter == 30: # 可以手動調時間
                    start_game += 1
                    counter = 0
            elif start_game == 5: # start game
                # score_strvar.set(f'Score: {score}, your life: {life}'); root.update()
                counter += 1
                minus_counter += 1
                # =============== 更新水果位置 ===================================
                for i in range(len(fruit)): 
                    rect = pygame.draw.rect(pygame_screen, (255, 255, 255), (fruit[i][1] - 5, fruit[i][0] - 5, 10, 10))
                    pygame_screen.blit(plus_ball, rect)
                    # pygame.draw.circle(pygame_screen, (255, 0, 0), (fruit[i][1], fruit[i][0]), 5, 0)
                    # cv2.circle(img, (fruit[i][1], fruit[i][0]), 5, (0, 0, 255), -1)
                    if counter % ((i + 1) * 100) == 0:
                        fruit[i] = [randint(30, img.shape[0] - 30), randint(30, img.shape[1] - 30)]
                if counter % (100 * (len(fruit) + 1)) == 0:
                    counter = 0
                # =============== 更新扣分位置 ===================================
                for i in range(len(minus_fruit)): 
                    rect = pygame.draw.rect(pygame_screen, (255, 255, 255), (minus_fruit[i][1] - 5, minus_fruit[i][0] - 5, 10, 10))
                    pygame_screen.blit(minus_ball, rect)
                    # pygame.draw.circle(pygame_screen, (0, 0, 100), (minus_fruit[i][1], minus_fruit[i][0]), 5, 0)
                    if minus_counter % ((i + 1) * 100) == 0:
                        minus_fruit[i] = [randint(30, img.shape[0] - 30), randint(30, img.shape[1] - 30)]
                if minus_counter % (100 * (len(minus_fruit) + 1)) == 0:
                    minus_counter = 0
                # ===============================================================
                pygame.draw.circle(pygame_screen, (0, 255, 0), (snack_position[1], snack_position[0]), 14 - game_level, 0)
                # cv2.circle(img, (snack_position[1], snack_position[0]), 14 - game_level, (0, 255, 0), -1)
                pygame_word(pygame_screen, f'Score: {score}, your life: {life}', (255, 100, 0), 28, 120, 60)
                if finger: # 更新目前手指方向座標，如果沒有就不更新(使用原方向)
                    (x1, y1) = ((hand_record[8][0] - hand_record[7][0]), 
                                    (hand_record[8][1] - hand_record[7][1]))
                snack_position = direct_snack(snack_position, x1, y1, game_level)
                if snack_position[0] < 0 or snack_position[0] > img.shape[0] or snack_position[1] < 0 or snack_position[1] > img.shape[1]:
                    life -= 1
                    if life == 0:
                        start_game += 1 # game over 
                        continue
                    else:
                        snack_position = [snack_position[0] % img.shape[0], snack_position[1] % img.shape[1]]
                for i in range(len(fruit)): # 檢查角色是否碰撞到獎勵
                    if not isExtrapolation(snack_position, 14 - game_level, fruit[i], 6):
                        score += 1 # 是 -> 分數加一
                        fruit[i] = [randint(30, img.shape[0] - 30), randint(30, img.shape[1] - 30)] # 重新變換位置
                for i in range(len(minus_fruit)): # 檢查角色是否碰撞到扣分
                    if not isExtrapolation(snack_position, 14 - game_level, minus_fruit[i], 6):
                        score -= 1 # 是 -> 分數扣一
                        minus_fruit[i] = [randint(30, img.shape[0] - 30), randint(30, img.shape[1] - 30)] # 重新變換位置
            else:
                print(f'Game Over, your score: {score}')
                pygame.quit()
                cv2.destroyAllWindows()
                # root.quit()
                exit()
            
            # update screen
            cv2.imshow('camera', cv2.resize(img, (int(img.shape[1] * 0.8), int(img.shape[0] * 0.8))))
            cv2.imshow('other camera', cv2.resize(img_other, (int(img_other.shape[1] * 0.8), int(img_other.shape[0] * 0.8))))
            pygame.display.update()
            pygame_screen.fill((255, 255, 255))
            # press key 'q'
            if cv2.waitKey(1) == ord('q'):
                pygame.quit()
                cv2.destroyAllWindows()
                # root.quit()
                break