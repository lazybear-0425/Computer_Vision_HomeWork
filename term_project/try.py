import cv2
import numpy as np
import mediapipe as mp
import math
import torch
import pandas
import tkinter as tk

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

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

def print_hand_action(hand):
    thumb = cal_angle((hand[0][0] - hand[2][0], hand[0][1] - hand[2][1]), (hand[3][0] - hand[4][0], hand[3][1] - hand[4][1]))
    fore_finger = cal_angle((hand[0][0] - hand[6][0], hand[0][1] - hand[6][1]), (hand[7][0] - hand[8][0], hand[7][1] - hand[8][1]))
    middle_finger = cal_angle((hand[0][0] - hand[9][0], hand[0][1] - hand[9][1]), (hand[11][0] - hand[12][0], hand[11][1] - hand[12][1]))
    ring_finger = cal_angle((hand[0][0] - hand[13][0], hand[0][1] - hand[13][1]), (hand[15][0] - hand[16][0], hand[15][1] - hand[16][1]))
    little_finger = cal_angle((hand[0][0] - hand[17][0], hand[0][1] - hand[17][1]), (hand[19][0] - hand[20][0], hand[19][1] - hand[20][1]))

    thres_angle = 50
    if (True) and (fore_finger >= thres_angle and thumb >= thres_angle and middle_finger >= thres_angle and ring_finger >= thres_angle and little_finger >= thres_angle):
        return 'zero'
    elif (fore_finger < thres_angle) and (thumb >= thres_angle and middle_finger >= thres_angle and ring_finger >= thres_angle and little_finger >= thres_angle):
        return 'one'
    elif (ring_finger < thres_angle and middle_finger < thres_angle) and (thumb >= thres_angle and fore_finger >= thres_angle and little_finger >= thres_angle):
        return 'two'
    elif (fore_finger < thres_angle and middle_finger < thres_angle) and (thumb >= thres_angle and ring_finger >= thres_angle and little_finger >= thres_angle):
        return 'two'
    elif (fore_finger < thres_angle and middle_finger < thres_angle and ring_finger < thres_angle) and (thumb >= thres_angle and little_finger >= thres_angle):
        return 'three'
    elif (fore_finger < thres_angle and middle_finger < thres_angle and ring_finger < thres_angle and little_finger < thres_angle) and (thumb >= thres_angle):
        return 'four'
    elif (fore_finger < thres_angle and middle_finger < thres_angle and ring_finger < thres_angle and little_finger < thres_angle and thumb < thres_angle) and (True):
        return 'five'
    elif (little_finger < thres_angle and thumb < thres_angle) and (fore_finger >= thres_angle and middle_finger >= thres_angle and ring_finger >= thres_angle):
        return 'six'
    elif (middle_finger < thres_angle and ring_finger < thres_angle and little_finger < thres_angle) and (fore_finger >= thres_angle and thumb >= thres_angle * 0.6):
        return 'ok' # 因為OK的拇指不明顯
    else: return ''

def show_tkinter(text):
    # tkinter
    root.title('detected objects')
    mytext = tk.Label(root, text=text)
    mytext.pack()
    root.update() # 使用 mainloop 會阻塞，導致無法使用

def begin_game(hand_record, img, thres_action):
    action = print_hand_action(hand_record)
    if action == '': return
    
    global actions
    if action != 'zero': actions.append(action)
    else: actions.clear() # reset
    
    if actions.count(action) % 10 == 0:
        print(f'action "{action}": \033[34m{actions.count(action)}\033[0m') # debug
    
    if actions.count('ok') >= thres_action:
        print('\033[33mOK~~~~~~~\033[0m')
        # YOLOv5
        # result = model(np.copy(img))
        # result.save()
        # store result in string
        show_tkinter('Game Start')
        return True
    return False

actions = []
start_game = False
level = 0
def detect_action(hand_record, img):
    global start_game
    thres_action = 40
    if not start_game:
        start_game = begin_game(hand_record, img, thres_action)
    else:
        if level == 0:
            cv2.putText(img, 'Please Choose Level 1-5', (img.shape[0] // 4, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    
#===============main======================================
cap = cv2.VideoCapture(0)
my_draw = mp.solutions.drawing_utils    # 繪圖方式
my_draw_style = mp.solutions.drawing_styles # 繪圖樣式
mp_hand = mp.solutions.hands
root = tk.Tk()

if not cap.isOpened():
    print('攝影機開啟失敗')
    exit()
    
with mp_hand.Hands(max_num_hands=1, min_detection_confidence=0.6) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            print('讀取失敗')
            exit()
        frame = cv2.flip(frame, 1)
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if result.multi_hand_landmarks:
            for hand_landmark in result.multi_hand_landmarks:
                hand_record = []
                for hand in hand_landmark.landmark:
                    hand_record.append([hand.x, hand.y]) # info of hand: x, y, z
                detect_action(hand_record, frame) # 印出目前手勢結果
                my_draw.draw_landmarks(frame, hand_landmark, mp_hand.HAND_CONNECTIONS, 
                                    landmark_drawing_spec=my_draw_style.get_default_pose_landmarks_style())
            '''
            result.multi_hand_landmarks
            ============================
            x, y, z
            len(result.multi_hand_landmarks) = 出現幾隻手
            ============================
            result.multi_handedness
            ============================
            index: 0(left), 1(right)
            score
            label: "Left", "Right"
            '''
        cv2.imshow('camera', frame)
        if cv2.waitKey(1) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()