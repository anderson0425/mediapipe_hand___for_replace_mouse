#mediapipe hand
import cv2
import mediapipe as mp
import pyautogui
import math
import time
import numpy as np
import math

camera_w, camera_h  = 0, 0# 相機拍攝的尺寸

TIME_Save_page, TIME_Save_As  = 0, 0 #用來讓跳前一頁和跳下一頁不會因為太敏感而連續執行太多次。
delay_Save_page, delay_Save_As = 5,5  #用來讓跳前一頁和跳下一頁不會因為太敏感而連續執行太多次。讓它間隔2秒
TIME_New_page, delay_New_page = 0, 3 #讓其不會因為太敏感而連續執行太多次。讓它間隔3秒
TIME_PrintScreen = 0 #用來讓螢幕截圖不會因為太敏感而連續執行太多次。
delay_PrintScreen = 0.5 #用來讓螢幕截圖不會因為太敏感而連續執行太多次。讓它間隔2秒
TIME_copy, TIME_paste = 0 , 0  #用來讓複製 貼上不會因為太敏感而連續執行太多次。
delay_copy, delay_paste= 1 , 3 #用來讓複製貼上不會因為太敏感而連續執行太多次。讓它間隔1秒
TIME_DOUBLE_CLICK ,TIME_CLICK =0,0

#以當前時間命名，並回傳。
# image name = a1 + a2 + "_" + a3 + a4 + "_"
def name_img_by_current_time():
    A = time.ctime()
    A = A.split()

    a1='0'
    mon = {'Oct':'10', 'Nov':'11' , 'Dec':'12', 'Jan':'01' , 'Feb':'02', 'Mar':'03' , 'Apr':'04', 'May':'05' , 'Jan':'06', 'Jul':'07' , 'Aug':'08', 'Sep':'09'}

    for (key, value) in mon.items():
        if A[1]==key:
            a1=value

    a2, a3, a4 = A[2], A[3][0:2], A[3][3:5]

    img_name_head = a1 + a2 + "_" + a3 + a4 + "_"
    return img_name_head

#p1,p2是某兩個節點的index
#r是圓的半徑。  因為要將中指及食指指尖的(x,y)座標畫圓
#(120,120,120)是那個圓的顏色
def distance(p1, p2, img, keypoint_pos, r=10, t=3):
    x1, y1 = int(keypoint_pos[p1][0]), int(keypoint_pos[p1][1])
    x2, y2 = int(keypoint_pos[p2][0]), int(keypoint_pos[p2][1])
    cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)

    #需要將參數轉整數，否則line()、circle()會報錯。
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), t)
    cv2.circle(img, (x1, y1), r, (120, 120, 120), cv2.FILLED)
    cv2.circle(img, (x2, y2), r, (120, 120, 120), cv2.FILLED)
    cv2.circle(img, (cx, cy), r, (255, 0, 255), cv2.FILLED)

    #算出兩個節點的連線長
    l = int(math.hypot(x2 - x1, y2 - y1))

    return l, img, [cx, cy]

#根據order做不同圖像處理
def img_processing(image, order):
    if order ==0: 
        po_image = cv2.flip(image, 1) ##水平翻轉

    if order==1:
        po_image = cv2.flip(image, 0) ##垂直翻轉

    if order==3: #rectangular_mask
        h, w, c = image.shape[0], image.shape[1], image.shape[2]
        w_mask = w/2
        image[:, 0:round(w_mask), :] = np.zeros((h, round(w_mask), c),np.uint8)
        po_image = image

    if order==4:  ## use gray level img  //onlY 2D, SO INVOKE ERROR    ValueError: could not broadcast input array from shape (255,255) into shape (255,255,3)  
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, pppo_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) ##二值化

        #use to correspond to spec. of API set_input_tensor, which needs 3D input, not 2D
        #SO I need to make gray level img into 3D, BUT LOOK SAME as 2D.
        po_image = np.zeros((pppo_image.shape[0], pppo_image.shape[1], 3))
        po_image[:,:,0] = pppo_image[:,:]
        po_image[:,:,1] = pppo_image[:,:]
        po_image[:,:,2] = pppo_image[:,:]

    if order==5: #set value(0, 255, 160, 80) in a manner + bilateralFilter blur to reduce Noise by  11 17 17
        h, w, c = image.shape[0], image.shape[1], image.shape[2]
        for k in range(0, c):
            for i in range(0, h, 3):
                for j in range(0, w, 3):
                    image[i, j, k] = 0
        for k in range(0, c):
            for i in range(0, h, 5):
                for j in range(0, w, 5):
                    image[i, j, k] = 255
        for k in range(0, c):
            for i in range(0, h, 4):
                for j in range(0, w, 7):
                    image[i, j, k] = 160
        for k in range(0, c):
            for i in range(0, h, 6):
                for j in range(0, w, 11):
                    image[i, j, k] = 80
        
        po_image = cv2.bilateralFilter(image, 11, 17, 17)  # smoothing filter

    if order==10:
        po_image = cv2.GaussianBlur(image, (5, 5), 0) #blur to reduce Noise

    if order==11:  #調全白
        w, h, c = image.shape[0], image.shape[1], image.shape[2]
        for i in range(w):
            for j in range(h):
                for k in range(c):
                    image[i,j,k] = 255
        po_image = image
    if order==12:  #調全黑
        w, h, c = image.shape[0], image.shape[1], image.shape[2]
        for i in range(w):
            for j in range(h):
                for k in range(c):
                    image[i,j,k] = 0
        po_image = image
    return po_image

#p1,p2是某兩個節點的index
#r是圓的半徑。  因為要將中指及食指指尖的(x,y)座標畫圓
#(120,120,120)是那個圓的顏色
def distance(p1, p2, img, keypoint_pos, r=10, t=3):
    x1, y1 = int(keypoint_pos[p1][0]), int(keypoint_pos[p1][1])
    x2, y2 = int(keypoint_pos[p2][0]), int(keypoint_pos[p2][1])
    cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)

    #需要將參數轉整數，否則line()、circle()會報錯。
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), t)
    cv2.circle(img, (x1, y1), r, (120, 120, 120), cv2.FILLED)
    cv2.circle(img, (x2, y2), r, (120, 120, 120), cv2.FILLED)
    cv2.circle(img, (cx, cy), r, (255, 0, 255), cv2.FILLED)

    #算出兩個節點的連線長
    l = int(math.hypot(x2 - x1, y2 - y1))

    return l, img, [cx, cy]

#把手勢放大
def enlarge(img, left_up, right_down):
    new_img = img.copy()
    size_w = abs(left_up[0] - right_down[0])
    size_h = abs(left_up[1] - right_down[1])
    
    # 得到正方形區域的image
    img_rec = img[left_up[1]:right_down[1], left_up[0]:right_down[0]]
    try:
        # 將此區域放大
        img_rec = cv2.resize(img_rec, (size_w*2, size_h*2), interpolation=cv2.INTER_CUBIC)
    except:
        return None
    
    center = [(left_up[0] + right_down[0])//2, (left_up[1] + right_down[1])//2] # [x, y]
    
    # 將原圖中正方形區域取代為放大後的圖
    for i in range(center[1]-size_h, center[1]+size_h):
        for j in range(center[0]-size_w, center[0]+size_w):
            try:
                new_img[i][j] = img_rec[i - (center[1]-size_h)][j - (center[0]-size_w)]
            except:
                continue
    
    return new_img 

# 針對給定的某個矩形做馬賽克
#輸入的left_up, right_down都是tuple，是某兩個邊界點的點(x,y)值。
#若要取得左上邊界點的y值，用left_up[1]。  若要取得右下邊界點的y值，用right_down[1]。
#若要取得左上邊界點的x值，用left_up[0]。  若要取得右下邊界點的x值，用right_down[0]。
#根據指令的範圍，將那一範圍弄為馬賽克，並回傳有馬賽克的圖片。
def mosaic(img, left_up, right_down):
    new_img = img.copy()

    # size代表此馬賽克區塊中每塊小區域的邊長
    #每個小分區為長寬(w,h)=(10,10)的小方塊。
    size = 10

    #根據左上和右下的y值以size為間隔去分區
    for i in range(left_up[1], right_down[1]-size-1, size):
        #根據左上和右下的X值以size為間隔去分區
        for j in range(left_up[0], right_down[0]-size-1, size):

            try:
                # 將此小區域中的每個像素都給定為最左上方的像素值，以此製造馬賽克。
                new_img[i:i + size, j:j + size] = img[i, j, :]
            except:
                pass

    return new_img

#vector_2d_angle的輸入是兩個型態為tuple的(x,y)
#v1,v2型態都是tuple
#v1是(x,y)，是某兩個節點的"x向量差"、"y向量差"。  v2也是(x,y)，是某兩個節點的"x向量差"、"y向量差"
#
# 求出v1,v2兩條向量的夾角
#vector_2d_angle回傳一個0-180之間的值，也就是輸入的兩個向量的夾角角度(以degree表示)。
def vector_2d_angle(v1,v2): 
    v1_x=v1[0] #某兩個節點的"x向量差"
    v1_y=v1[1] #某兩個節點的"y向量差"
    v2_x=v2[0]
    v2_y=v2[1]

    try:
        #(v1_x**2 + v1_y**2)**0.5是某兩個節點的直線差(而且必為非負數)。  (v2_x**2 + v2_y**2)**0.5也是。
        angle_= math.degrees(math.acos((v1_x*v2_x + v1_y*v2_y)/(((v1_x**2 + v1_y**2)**0.5)*((v2_x**2 + v2_y**2)**0.5))))
    except:
        angle_ = 100000.
    return angle_

#輸入是一個長度21的list，為那21個節點分別的(x,y)值
#hand_裡面:   [2]....[4]是大拇指 / [5]....[8]是食指 / [9]....[12]是中指 / [13]...[16]是無名指 / [17]...[20]是小指
#hand_[i]是index=i的節點
#hand_[i][0]是index=i的節點的X值，hand_[i][1]是index=i的節點的Y值
#
#回傳一個list包含大拇指、食指、中指、無名指、小指這5根手指個別的角度
#個別算每個手指的夾角，應該是因為每根手指的夾角都可以視為二維空間的角度去處理。
def hand_angle(hand_):

    #會儲存每根手指上面的夾角角度
    #因此會有大拇指、食指、中指、無名指、小指這5根手指的角度
    angle_list = []

    #---------------------------- thumb 大拇指角度
    #vector_2d_angle的輸入是兩個型態為tuple的(x,y)
    #vector_2d_angle回傳一個0-180之間的值，也就是輸入的兩個向量的夾角角度(以degree表示)。
    #此angle為 "node[0]與node[2]連線形成的向量" 與 "node[3]與node[4]連線形成的向量" 的夾角角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[2][0])),(int(hand_[0][1])-int(hand_[2][1]))),  #這個(x,y) = (x0-x2, y0-y2)
        ((int(hand_[3][0])- int(hand_[4][0])),(int(hand_[3][1])- int(hand_[4][1])))  #這個(x,y) = (x3-x4, y3-y4)
    )
    angle_list.append(angle_)
    #-------------------------------------------------

    #---------------------------- index 食指角度
    #此angle為 "node[0]與node[6]連線形成的向量" 與 "node[7]與node[8]連線形成的向量" 的夾角角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])-int(hand_[6][0])),(int(hand_[0][1])- int(hand_[6][1]))),
        ((int(hand_[7][0])- int(hand_[8][0])),(int(hand_[7][1])- int(hand_[8][1])))
    )
    angle_list.append(angle_)
    #-------------------------------------------------

    #---------------------------- middle 中指角度
    #此angle為 "node[0]與node[10]連線形成的向量" 與 "node[11]與node[12]連線形成的向量" 的夾角角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[10][0])),(int(hand_[0][1])- int(hand_[10][1]))),
        ((int(hand_[11][0])- int(hand_[12][0])),(int(hand_[11][1])- int(hand_[12][1])))
    )
    angle_list.append(angle_)
    #-------------------------------------------------

    #---------------------------- ring 無名指角度
    #此angle為 "node[0]與node[14]連線形成的向量" 與 "node[15]與node[16]連線形成的向量" 的夾角角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[14][0])),(int(hand_[0][1])- int(hand_[14][1]))),
        ((int(hand_[15][0])- int(hand_[16][0])),(int(hand_[15][1])- int(hand_[16][1])))
    )
    angle_list.append(angle_)
    #-------------------------------------------------

    #---------------------------- pink 小拇指角度
    #此angle為 "node[0]與node[18]連線形成的向量" 與 "node[19]與node[20]連線形成的向量" 的夾角角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[18][0])),(int(hand_[0][1])- int(hand_[18][1]))),
        ((int(hand_[19][0])- int(hand_[20][0])),(int(hand_[19][1])- int(hand_[20][1])))
    )
    angle_list.append(angle_)
    #-------------------------------------------------

    #大拇指、食指、中指、無名指、小指這5根手指的角度
    return angle_list

# 輸入的angle_list是一個長度為5，元素值介於0到180的list，裡面包含5根手指個別的夾角
#angle_list: [0]-->大拇指 / [1]-->食指 / [2]-->中指 / [3]-->無名指 / [4]-->小指
#
# 當只有伸出食指時，回傳一個string，內有"index finger up!"。  
# 當手勢不是伸出食指，則回傳none。
def hand_gesture(angle_list, keypoint_pos):

    gesture_str = None #empty string

    if 100000. not in angle_list:
        #零根手指
        ##沒有伸出任何手指 (ROCK石頭)
        if (angle_list[0]>90) and (angle_list[1]>90) and (angle_list[2]>90) and (angle_list[3]>90) and (angle_list[4]>90):
            gesture_str = "0"   
            return gesture_str



        #一根手指
        #只伸出大拇指，且朝上 (比讚)
        if (angle_list[0]<30) and (angle_list[1]>40) and (angle_list[2]>40) and (angle_list[3]>40) and (angle_list[4]>40) and keypoint_pos[0][1]>keypoint_pos[4][1]:
            gesture_str = "1UP"
            return gesture_str

        #只伸出大拇指，且朝下 (倒讚)
        if (angle_list[0]<30) and (angle_list[1]>40) and (angle_list[2]>40) and (angle_list[3]>40) and (angle_list[4]>40) and keypoint_pos[0][1]<keypoint_pos[4][1]:
            gesture_str = "1DOWN"
            return gesture_str

        #只伸出食指
        if (angle_list[0]>40) and (angle_list[1]<30) and (angle_list[2]>40) and (angle_list[3]>40) and (angle_list[4]>40):
            gesture_str = "2"
            return gesture_str

        #只伸出中指
        if (angle_list[0]>40) and (angle_list[1]>40) and (angle_list[2]<30) and (angle_list[3]>40) and (angle_list[4]>40):
            gesture_str = "3"
            return gesture_str

        #只伸出無名指
        if (angle_list[0]>40) and (angle_list[1]>40) and (angle_list[2]>40) and (angle_list[3]<30) and (angle_list[4]>40):
            gesture_str = "4"
            return gesture_str

        #只伸出小指
        if (angle_list[0]>40) and (angle_list[1]>40) and (angle_list[2]>40) and (angle_list[3]>40) and (angle_list[4]<30):
            gesture_str = "5"
            return gesture_str


        #兩根手指
#        #只伸出大拇指、食指
#        if (angle_list[0]<30) and (angle_list[1]<30) and (angle_list[2]>40) and (angle_list[3]>40) and (angle_list[4]>40):
#            gesture_str = "12"

        #只伸出大拇指、食指
        #一個槍的手勢，朝左(根據大拇指及食指指尖的x座標相對大小判斷)
        if (angle_list[0]<30) and (angle_list[1]<30) and (angle_list[2]>40) and (angle_list[3]>40) and (angle_list[4]>40) and keypoint_pos[4][0]>keypoint_pos[8][0]:
            gesture_str = "12_left"
            return gesture_str

        #只伸出大拇指、食指
        #一個槍的手勢，朝右(根據大拇指及食指指尖的x座標相對大小判斷)
        if (angle_list[0]<30) and (angle_list[1]<30) and (angle_list[2]>40) and (angle_list[3]>40) and (angle_list[4]>40) and keypoint_pos[4][0]<keypoint_pos[8][0]:
            gesture_str = "12_right"
            return gesture_str
        
        #只伸出大拇指、中指
        if (angle_list[0]<30) and (angle_list[1]>40) and (angle_list[2]<30) and (angle_list[3]>40) and (angle_list[4]>40):
            gesture_str = "13"
            return gesture_str
        
        #只伸出大拇指、小指
        if (angle_list[0]<30) and (angle_list[1]>40) and (angle_list[2]>40) and (angle_list[3]>40) and (angle_list[4]<30):
            gesture_str = "15"
            return gesture_str
        
        #只伸出食指、中指 (YA、SCISSOR剪刀)
        if (angle_list[0]>40) and (angle_list[1]<30) and (angle_list[2]<30) and (angle_list[3]>40) and (angle_list[4]>40):
            gesture_str = "23"
            return gesture_str
        
        #只伸出食指、小指
        if (angle_list[0]>40) and (angle_list[1]<30) and (angle_list[2]>40) and (angle_list[3]>40) and (angle_list[4]<30):
            gesture_str = "25"
            return gesture_str
        
        #只伸出中指、無名指
        if (angle_list[0]>40) and (angle_list[1]>40) and (angle_list[2]<30) and (angle_list[3]<30) and (angle_list[4]>40):
            gesture_str = "34"
            return gesture_str
        
        #只伸出中指、小指
        if (angle_list[0]>40) and (angle_list[1]>40) and (angle_list[2]<30) and (angle_list[3]>40) and (angle_list[4]<30):
            gesture_str = "35"
            return gesture_str
        
        #只伸出無名指、小指
        if (angle_list[0]>40) and (angle_list[1]>40) and (angle_list[2]>40) and (angle_list[3]<30) and (angle_list[4]<30):
            gesture_str = "45"
            return gesture_str


        #三根手指
        #只伸出大拇指、食指、中指
        if (angle_list[0]<15) and (angle_list[1]<30) and (angle_list[2]<30) and (angle_list[3]>40) and (angle_list[4]>40):
            gesture_str = "123"
            return gesture_str
        
        #只伸出大拇指、食指、小指  (蜘蛛人吐絲手勢)
        if (angle_list[0]<30) and (angle_list[1]<30) and (angle_list[2]<30) and (angle_list[3]>40) and (angle_list[4]<30):
            gesture_str = "125"
            return gesture_str
        
        #只伸出食指、中指、無名指
        if (angle_list[0]>40) and (angle_list[1]<30) and (angle_list[2]<30) and (angle_list[3]<30) and (angle_list[4]>40):
            gesture_str = "234"
            return gesture_str
        
        #只伸出食指、中指、小指
        if (angle_list[0]>40) and (angle_list[1]<30) and (angle_list[2]<30) and (angle_list[3]>40) and (angle_list[4]<30):
            gesture_str = "235"
            return gesture_str
        
        #只伸出中指、無名指、小指
        if (angle_list[0]>40) and (angle_list[1]>40) and (angle_list[2]<30) and (angle_list[3]<30) and (angle_list[4]<30):
            gesture_str = "345"
            return gesture_str
        

        #四根手指
        #只伸出大拇指、食指、中指、小指
        if (angle_list[0]<30) and (angle_list[1]<30) and (angle_list[2]<30) and (angle_list[3]>40) and (angle_list[4]<30):
            gesture_str = "1235"
            return gesture_str
        
        #只伸出大拇指、食指、無名指、小指
        if (angle_list[0]<30) and (angle_list[1]<30) and (angle_list[2]>40) and (angle_list[3]<30) and (angle_list[4]<30):
            gesture_str = "1245"
            return gesture_str
        
        #只伸出食指、中指、無名指、小指
        if (angle_list[0]>40) and (angle_list[1]<30) and (angle_list[2]<30) and (angle_list[3]<30) and (angle_list[4]<30):
            gesture_str = "2345"
            return gesture_str
        

        #五根手指
        #所有手指都伸出來 (PAPER布)
        if (angle_list[0]<30) and (angle_list[1]<30) and (angle_list[2]<30) and (angle_list[3]<30) and (angle_list[4]<30):
            gesture_str = "12345"
            return gesture_str


        #比較難比的手勢
        #14  24  124  1234
        #只伸出大拇指、無名指
        if (angle_list[0]<30) and (angle_list[1]>40) and (angle_list[2]>40) and (angle_list[3]<30) and (angle_list[4]>40):
            gesture_str = "14"
            return gesture_str
        
        #只伸出食指、無名指
        if (angle_list[0]>40) and (angle_list[1]<30) and (angle_list[2]>40) and (angle_list[3]<30) and (angle_list[4]>40):
            gesture_str = "24"
            return gesture_str
        
        #只伸出大拇指、食指、無名指
        if (angle_list[0]<30) and (angle_list[1]<30) and (angle_list[2]>40) and (angle_list[3]<30) and (angle_list[4]>40):
            gesture_str = "124"
            return gesture_str
        
        #只伸出大拇指、食指、中指、無名指
        if (angle_list[0]<30) and (angle_list[1]<30) and (angle_list[2]<30) and (angle_list[3]<30) and (angle_list[4]>40):
            gesture_str = "1234"
            return gesture_str

    return gesture_str

def ratio_grow(RATIO, add=0.15):
    RATIO_TMP = float("{:.2f}".format(RATIO))  #取到小數點後第2位
    return RATIO_TMP + add

#會跟隨手部的滑鼠游標
def mouse_track(frame, keypoint_pos):
    length, frame, lineInfo = distance(8, 12,frame, keypoint_pos) #8和12是食指和中指指尖的index

    #當兩指指尖距離很接近，才會做滑鼠控制
    #則螢幕會顯示一個綠色的點
    if length < 30:
        cv2.circle(frame, (lineInfo[0], lineInfo[1]),15, (0, 255, 0), cv2.FILLED)

        #mapping方式---------------------------------------------------------------
        #當視窗在左上角時，指尖座標範圍是 x = 0 ~ 1240, y = 0 ~ 700
        #而這台電腦螢幕長寬是 WIDTH = 1920, HEIGHT = 1080
        #
        #因此要將這個從原本的指尖座標mapping到螢幕的滑鼠座標。
        # X = 1920/1240 * x
        # Y = 1080/700 * y
        #--------------------------------------------------------------------------

        #把座標移動螢幕正中央:   
        # X2=X1  620  350------(DX,DY)=(+340,+190)------->960  540
        # (LX, RX) = (960-620, 960+620) = (340, 1580)
        # (LY, RY) = (540-350, 540+350) = (190, 890)
        #
        #移動正中央後，這個小視窗的邊界及中央座標各為:
        # 左上 (LX, LY) = (340, 190)   ----> (0, 0)   340-(960-340)*(340/620)=0   190-(540-190)*(190/350)=0
        # (CX, CY) = (960, 540)  ----> (960, 540)
        # 右下 (RX, RY) = (1580, 890)  -----> (1920, 1080)
        # 左下 (ldx, ldy) = (340, 890) ----> (0, 1080)
        # 右上 (rux, ruy) = (1580, 190)  -----> (1920, 0)
        #
        #把這個小視窗的邊界及中央座標MAPPING到大視窗(電腦螢幕尺寸)，則座標變成:
        #小視窗半寬960-340=620，小視窗半高540-190=350
        #大視窗半寬1920/2=960，大視窗半高1080/2=540
        #小視窗的(x,y)=(150,200)---移到中央---->(x,y)=(150+340, 200+190)=(490, 390)------拓展到大螢幕----->(x,y)=(490*[], 390+)

        PLACE_INDEX_FINGER = ""  #紀錄指尖移動中央後，其座標相對於大螢幕中央在哪個方向(左上 右下 左下  右上)

        #食指指尖的座標是把視窗放在左上角時的座標
        IndexFinger_x, IndexFinger_y = keypoint_pos[8][0], keypoint_pos[8][1] #食指指尖的x,y座標

        W_LITTLE , H_LITTLE = 1240 , 700  # FIXME: 實測出來，要根據攝像頭尺寸去改!!
        W_BIG , H_BIG = pyautogui.size()  #這個電腦螢幕尺寸
        half_W_LITTLE , half_H_LITTLE = W_LITTLE/2 , H_LITTLE/2  #小視窗半寬  小視窗半高
        half_W_BIG , half_H_BIG = W_BIG/2 , H_BIG/2  #大視窗半寬   大視窗半高
        LX , LY = (W_BIG/2 - W_LITTLE/2) , (H_BIG/2 - H_LITTLE/2)  #左上
        RX , RY = (W_BIG/2 + W_LITTLE/2) , (H_BIG/2 + H_LITTLE/2)  #右下
        CX , CY = LX + W_LITTLE / 2 , LY + H_LITTLE / 2  #中央

        L_BIG = int(math.hypot(W_BIG, H_BIG))  #大視窗的對角線長
        L_LITTLE = int(math.hypot(W_LITTLE, H_LITTLE))  #小視窗的對角線長
        RATIO = L_BIG/L_LITTLE #兩個視窗的對角線的比率  #1.5474349964862966

        #為了處理邊界問題: 使用者必須讓手指在小視窗的位置極為靠近小視窗邊界，才能讓大視窗的滑鼠挪到大視窗的邊界。
        # 這樣常常讓使用者的手跑出小視窗，導致沒有完整的手在小視窗哩，導致無法手勢辨識。
        #因此藉由微微提高縮放比率，
        # 讓小視窗的座標不必很貼近小視窗邊界就能MAPPING出超越大視窗邊界的大視窗座標。
        RATIO = ratio_grow(RATIO, add=0.4)

        #把小螢幕座標印射到大螢幕的正中央
        IndexFinger_x, IndexFinger_y = IndexFinger_x + LX , IndexFinger_y + LY

        #判斷相對於中央在哪個方向(左上 右下 左下  右上)
        if IndexFinger_x==CX and IndexFinger_y==CY:
            PLACE_INDEX_FINGER="middle" #正中央
        else:  #若不在正中央
            if IndexFinger_x>CX:
                if IndexFinger_y>CY:
                    PLACE_INDEX_FINGER="right_down"  #右下
                else:
                    PLACE_INDEX_FINGER="right_up"  #右上
            if IndexFinger_x<CX:
                if IndexFinger_y>CY:
                    PLACE_INDEX_FINGER="left_down"  #左下
                else:
                    PLACE_INDEX_FINGER="left_up"  #左上     

        #把小螢幕的x,y印射到大螢幕的x,y
        if PLACE_INDEX_FINGER=="right_down":  #右下
            dx_little = IndexFinger_x - CX
            dy_little = IndexFinger_y - CY
            dx_big = dx_little * RATIO
            dy_big = dy_little * RATIO
            AFTER_IndexFinger_x = CX + dx_big
            AFTER_IndexFinger_y = CY + dy_big

        elif PLACE_INDEX_FINGER=="right_up":  #右上
            dx_little = IndexFinger_x - CX
            dy_little = IndexFinger_y - CY
            dx_big = dx_little * RATIO
            dy_big = dy_little * RATIO
            AFTER_IndexFinger_x = CX + dx_big
            AFTER_IndexFinger_y = CY + dy_big

        elif PLACE_INDEX_FINGER=="left_down":  #左下
            dx_little = IndexFinger_x - CX
            dy_little = IndexFinger_y - CY
            dx_big = dx_little * RATIO
            dy_big = dy_little * RATIO
            AFTER_IndexFinger_x = CX + dx_big
            AFTER_IndexFinger_y = CY + dy_big

        elif PLACE_INDEX_FINGER=="left_up":  #左上     
            dx_little = IndexFinger_x - CX
            dy_little = IndexFinger_y - CY
            dx_big = dx_little * RATIO
            dy_big = dy_little * RATIO
            AFTER_IndexFinger_x = CX + dx_big
            AFTER_IndexFinger_y = CY + dy_big
        else:
            print("error")

        #處理寬高的overflow，讓它飽和在邊界
        if AFTER_IndexFinger_x >W_BIG:
            AFTER_IndexFinger_x=W_BIG-2
        if AFTER_IndexFinger_y>H_BIG:
            AFTER_IndexFinger_y=H_BIG-2

        #print(AFTER_IndexFinger_x)
        #print(AFTER_IndexFinger_y)

        #根據mapping後的座標去移動滑鼠
        #記得把小視窗放大螢幕正中央，這樣才看得出效果!!
        pyautogui.moveTo(AFTER_IndexFinger_x, AFTER_IndexFinger_y) 

        #強調指尖的位置
        #但因為小視窗和大視窗的差異，這個反而會無法正確顯示。因為值是大視窗坐標系的值，但它卻是在小視窗中顯示。
        #cv2.circle(frame, (int(IndexFinger_x), int(IndexFinger_y)), 15, (120, 0, 120), cv2.FILLED)

        #強調指尖的位置
        #基於上面的原因，因此不能用IndexFinger_x, IndexFinger_y
        cv2.circle(frame, (int(keypoint_pos[8][0]), int(keypoint_pos[8][1])), 15, (120, 0, 120), cv2.FILLED)

    return frame

#滑鼠拖曳
#要像滑鼠控制一樣做mapping才行
def mouse_drag(frame, keypoint_pos):
    length, frame, lineInfo = distance(8, 12,frame, keypoint_pos) #8和12是食指中指指尖的index。  lineInfo是兩指尖連線的中點位置

    #當兩指指尖距離很接近，才會做滑鼠控制
    if length < 50:
        cv2.circle(frame, (lineInfo[0], lineInfo[1]),15, (0, 120, 0), cv2.FILLED)

        #mapping方式---------------------------------------------------------------
        #當視窗在左上角時，指尖座標範圍是 x = 0 ~ 1240, y = 0 ~ 700
        #而這台電腦螢幕長寬是 WIDTH = 1920, HEIGHT = 1080
        #
        #因此要將這個從原本的指尖座標mapping到螢幕的滑鼠座標。
        # X = 1920/1240 * x
        # Y = 1080/700 * y
        #--------------------------------------------------------------------------

        #把座標移動螢幕正中央:   
        # X2=X1  620  350------(DX,DY)=(+340,+190)------->960  540
        # (LX, RX) = (960-620, 960+620) = (340, 1580)
        # (LY, RY) = (540-350, 540+350) = (190, 890)
        #
        #移動正中央後，這個小視窗的邊界及中央座標各為:
        # 左上 (LX, LY) = (340, 190)   ----> (0, 0)   340-(960-340)*(340/620)=0   190-(540-190)*(190/350)=0
        # (CX, CY) = (960, 540)  ----> (960, 540)
        # 右下 (RX, RY) = (1580, 890)  -----> (1920, 1080)
        # 左下 (ldx, ldy) = (340, 890) ----> (0, 1080)
        # 右上 (rux, ruy) = (1580, 190)  -----> (1920, 0)
        #
        #把這個小視窗的邊界及中央座標MAPPING到大視窗(電腦螢幕尺寸)，則座標變成:
        #小視窗半寬960-340=620，小視窗半高540-190=350
        #大視窗半寬1920/2=960，大視窗半高1080/2=540
        #小視窗的(x,y)=(150,200)---移到中央---->(x,y)=(150+340, 200+190)=(490, 390)------拓展到大螢幕----->(x,y)=(490*[], 390+)

        PLACE_INDEX_FINGER = ""  #紀錄指尖移動中央後，其座標相對於大螢幕中央在哪個方向(左上 右下 左下  右上)

        #中指指尖的座標是把視窗放在左上角時的座標
        IndexFinger_x, IndexFinger_y = keypoint_pos[12][0], keypoint_pos[12][1] #中指指尖的x,y座標

        W_LITTLE , H_LITTLE = 1240 , 700 #FIXME: 實測出來，要根據攝像頭尺寸去改!!
        W_BIG , H_BIG = pyautogui.size()  #這個電腦螢幕尺寸
        half_W_LITTLE , half_H_LITTLE = W_LITTLE/2 , H_LITTLE/2  #小視窗半寬  小視窗半高
        half_W_BIG , half_H_BIG = W_BIG/2 , H_BIG/2  #大視窗半寬   大視窗半高
        LX , LY = (W_BIG/2 - W_LITTLE/2) , (H_BIG/2 - H_LITTLE/2)  #左上
        RX , RY = (W_BIG/2 + W_LITTLE/2) , (H_BIG/2 + H_LITTLE/2)  #右下
        CX , CY = LX + W_LITTLE / 2 , LY + H_LITTLE / 2  #中央

        L_BIG = int(math.hypot(W_BIG, H_BIG))  #大視窗的對角線長
        L_LITTLE = int(math.hypot(W_LITTLE, H_LITTLE))  #小視窗的對角線長
        RATIO = L_BIG/L_LITTLE #兩個視窗的對角線的比率  #1.5474349964862966

        #為了處理邊界問題: 使用者必須讓手指在小視窗的位置極為靠近小視窗邊界，才能讓大視窗的滑鼠挪到大視窗的邊界。
        # 這樣常常讓使用者的手跑出小視窗，導致沒有完整的手在小視窗哩，導致無法手勢辨識。
        #因此藉由微微提高縮放比率，
        # 讓小視窗的座標不必很貼近小視窗邊界就能MAPPING出超越大視窗邊界的大視窗座標。
        RATIO = ratio_grow(RATIO, add=10.0)

        #把小螢幕座標印射到大螢幕的正中央
        IndexFinger_x, IndexFinger_y = IndexFinger_x + LX , IndexFinger_y + LY

        #判斷相對於中央在哪個方向(左上 右下 左下  右上)
        if IndexFinger_x==CX and IndexFinger_y==CY:
            PLACE_INDEX_FINGER="middle" #正中央
        else:  #若不在正中央
            if IndexFinger_x>CX:
                if IndexFinger_y>CY:
                    PLACE_INDEX_FINGER="right_down"  #右下
                else:
                    PLACE_INDEX_FINGER="right_up"  #右上
            if IndexFinger_x<CX:
                if IndexFinger_y>CY:
                    PLACE_INDEX_FINGER="left_down"  #左下
                else:
                    PLACE_INDEX_FINGER="left_up"  #左上     

        #把小螢幕的x,y印射到大螢幕的x,y
        if PLACE_INDEX_FINGER=="right_down":  #右下
            dx_little = IndexFinger_x - CX
            dy_little = IndexFinger_y - CY
            dx_big = dx_little * RATIO
            dy_big = dy_little * RATIO
            AFTER_IndexFinger_x = CX + dx_big
            AFTER_IndexFinger_y = CY + dy_big

        elif PLACE_INDEX_FINGER=="right_up":  #右上
            dx_little = IndexFinger_x - CX
            dy_little = IndexFinger_y - CY
            dx_big = dx_little * RATIO
            dy_big = dy_little * RATIO
            AFTER_IndexFinger_x = CX + dx_big
            AFTER_IndexFinger_y = CY + dy_big

        elif PLACE_INDEX_FINGER=="left_down":  #左下
            dx_little = IndexFinger_x - CX
            dy_little = IndexFinger_y - CY
            dx_big = dx_little * RATIO
            dy_big = dy_little * RATIO
            AFTER_IndexFinger_x = CX + dx_big
            AFTER_IndexFinger_y = CY + dy_big

        elif PLACE_INDEX_FINGER=="left_up":  #左上     
            dx_little = IndexFinger_x - CX
            dy_little = IndexFinger_y - CY
            dx_big = dx_little * RATIO
            dy_big = dy_little * RATIO
            AFTER_IndexFinger_x = CX + dx_big
            AFTER_IndexFinger_y = CY + dy_big
        else:
            print("error")

        #處理寬高的overflow，讓它飽和在邊界
        if AFTER_IndexFinger_x >W_BIG:
            AFTER_IndexFinger_x=W_BIG-1
        if AFTER_IndexFinger_y>H_BIG:
            AFTER_IndexFinger_y=H_BIG-1

        #根據mapping後的座標去拖曳檔案或資料夾(拖曳是由"一直按住滑鼠左鍵，再移動"去實現的)
        #記得把小視窗放大螢幕正中央，這樣才看得出效果!!
        pyautogui.dragTo(AFTER_IndexFinger_x, AFTER_IndexFinger_y, button='left', duration=1)
        #pyautogui.dragTo(AFTER_IndexFinger_x, AFTER_IndexFinger_y, button='left')

        #強調指尖的位置
        #但因為小視窗和大視窗的差異，這個反而會無法正確顯示。因為值是大視窗坐標系的值，但它卻是在小視窗中顯示。
        #cv2.circle(frame, (int(IndexFinger_x), int(IndexFinger_y)), 15, (120, 0, 120), cv2.FILLED)

        #強調指尖的位置
        #基於上面的原因，因此不能用IndexFinger_x, IndexFinger_y
        cv2.circle(frame, (int(keypoint_pos[12][0]), int(keypoint_pos[12][1])), 15, (120, 0, 120), cv2.FILLED)

    return frame

#根據手勢，做出不同操作。
def BEHAVIOR(frame, gesture_str, keypoint_pos, angle_list, bounding):

    global TIME_Save_page, TIME_Save_As #讓其不會因為太敏感而連續執行太多次。
    global delay_Save_As, delay_Save_page #讓其不會因為太敏感而連續執行太多次。讓它間隔2秒
    global TIME_New_page, delay_New_page #讓其不會因為太敏感而連續執行太多次。讓它間隔2秒
    global TIME_PrintScreen #用來讓螢幕截圖不會因為太敏感而連續執行太多次。
    global delay_PrintScreen #用來讓螢幕截圖不會因為太敏感而連續執行太多次。讓它間隔2秒
    global TIME_copy, TIME_paste #用來讓複製 貼上不會因為太敏感而連續執行太多次。
    global delay_copy, delay_paste #用來讓複製貼上不會因為太敏感而連續執行太多次。讓它間隔2秒
    global TIME_DOUBLE_CLICK ,TIME_CLICK

    ##當只有食指和中指伸出，且兩指併冗(距離夠小)，才可以操控滑鼠---->這是為了避免只用食指偵測時，會因為收起食指而讓滑鼠位置飄移
    #記得別使使用"123"、"235"、"2"、"0"，
    # 因為可能操控滑鼠時，會不小心因為光害或其他的因素，"23"被誤認為這些手勢。
    if gesture_str == "23":
        gesture_str = gesture_str + "_mouse control"
        frame = mouse_track(frame, keypoint_pos)

    #當2345，則滑鼠點擊
    #每次點擊間隔3秒
    if gesture_str == "2":
        LATE_TIME_CLICK = time.time()
        if LATE_TIME_CLICK - TIME_CLICK > 2 or TIME_CLICK==0:  #讓每次單擊都間隔3秒
            gesture_str = gesture_str + "_click!!!"
            TIME_CLICK = time.time()
            #讓滑鼠做出點擊的動作
            pyautogui.click()
        else:
            gesture_str = gesture_str + "_no_click"

    #當12345時，則滑鼠雙擊
    if gesture_str == "45":
        LATE_TIME_DOUBLE_CLICK = time.time()
        if LATE_TIME_DOUBLE_CLICK - TIME_DOUBLE_CLICK > 3 or TIME_DOUBLE_CLICK==0:  #讓每次單擊都間隔3秒
            gesture_str = gesture_str + "_double_click!!!"
            TIME_DOUBLE_CLICK = time.time()
            #讓滑鼠做出雙重點擊的動作
            pyautogui.doubleClick()
        else:
            gesture_str = gesture_str + "_no_double_click"

    ##當只有12伸出，朝左，則複製(每次需間隔delay_Save_page秒)
    if gesture_str == "12_left":
        LATE_TIME_Save_page = time.time()
        if LATE_TIME_Save_page - TIME_Save_page > delay_Save_page or TIME_Save_page==0:  #讓每次單擊都間隔delay_Save_page秒
            gesture_str = gesture_str[:2] + "_copy"
            TIME_Save_page = time.time()

            #做出複製的動作
            
#    ##當只有12伸出，朝右，則貼上(每次需間隔delay_Save_As秒)
#    if gesture_str == "12_right":
#        LATE_TIME_Save_As = time.time()
#        if LATE_TIME_Save_As - TIME_Save_As > delay_Save_As or TIME_Save_As==0:  #讓每次單擊都間隔delay_Save_As秒
#            gesture_str = gesture_str[:2] + "_paste"
#            TIME_Save_As = time.time()
#
#            #做出貼上的動作
#            pyautogui.hotkey('shift', 'insert')
#
    ## 當123，則螢幕截圖(win+PrintScreen)
    if gesture_str == "123":
        LATE_TIME_PrintScreen = time.time()
        if LATE_TIME_PrintScreen - TIME_PrintScreen > delay_PrintScreen or TIME_PrintScreen==0:  #讓每次單擊都間隔delay_Next_page秒
            gesture_str = gesture_str + "_PrintScreen"
            TIME_PrintScreen = time.time()

            #做出螢幕截圖的動作
            pyautogui.hotkey('win', 'printscreen')

    #當只有234伸出，且23很近，則滑鼠拖曳
    if gesture_str == "5":
        gesture_str = gesture_str + "_drag"
        pyautogui.hotkey('win', 'ctrl', 'o')

    return frame, gesture_str

def detect():
    global camera_w, camera_h # 相機拍攝的尺寸

    gesture_dict_for_img_processing = {
        "Nike": None,
        "thumb up!": 11, 
        "index finger up!": None, 
        "middle finger up!": None, 
        "ring finger up!": None, 
        "little finger up!": None
    }
    mp_hands = mp.solutions.hands

    #1. static_image_mode=False ---> If set to false, the solution treats the input images as a video stream. It will try to detect hands in the first input images
    #1. static_image_mode=true ---> If set to true, hand detection runs on every input image
    #2. max_num_hands=1 ---> Maximum number of hands (1) to detect.
    hands = mp_hands.Hands(
            static_image_mode=False, #只會偵測第一張(但設為true又不知道會出什麼狀況，可能會變慢吧?)，一般設為false就好。
            max_num_hands=1,  #可以偵測到的手的最大數量。(預設為2隻)
            min_detection_confidence=0.75,  #手勢辨識後，會有一個confidence(test accuracy)。 而這個參數表示手勢辨識的評估值(test accuracy)必須大於這個參數值，(才視作成功的辨識)才會採用這個辨識結果。
            min_tracking_confidence=0.75  #目標追蹤，但這個我不知道有什麼效果。(若static_image_mode=true 則這個參數可以忽略。)必須大於這個參數值，才視作成功的追蹤。
    )

    # 開啟視訊鏡頭讀取器
    #並且指定使用index=0的攝像頭(如果同時接很多個攝像頭，就要設定不同value)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    #設定frame的寬高 = (width, height)
    #pyautogui.size()回傳螢幕尺寸
    width,height = pyautogui.size()  #1536, 864  恰好是camera拍攝尺寸
    camera_w, camera_h = width,height  #作為global

    cap.set(3,width) #表示將width給予frame的寬
    cap.set(4,height)#表示將height給予frame的高

    #會一直執行 "拍一張拍照，然後對照片做處理" 的動作。
    while True:
        
        i_time = time.time() #紀錄一開始的時間

        str_for_put_text = ""

        # 偵測影像中的手部
        #frame是一張照片(frame這個詞專門用來稱呼影片中的其中一張照片-----影片是由一張張照片構成，只是速度太快，眼睛覺得它是連續的。)
        #拍一張照片
        success, frame = cap.read()  #拍攝照片

        #若沒有成功開啟相機模組，則跳過這次的iteration
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  #轉rgb
        frame= cv2.flip(frame,1)  #左右翻轉

        #因為mediapipe需要的輸入圖片格式為rgb
        results = hands.process(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)#為了符合imshow()所要求的輸入圖片形式: bgr

#        print(results.multi_hand_landmarks)  #如果畫面有手，則會回傳一個list，裡面存有很多的型態為dict的landmark。(landmark裡面有x,y,z的數值，皆為-1到1之間)  若沒有手，則回傳none

#        print(results.multi_handedness)  #判斷是左手或右手(可用其輸出的index做之後判斷)，還有這個辨識結果的準確率
#        # 若畫面有手，則回傳一個classification object，裡面有index, score, label。  
#        #       若是左手，則index=0，且label為"left"，score則輸出"是左手的confidency(左手右手--classification)"。
#        #       若是右手，則index=1，且label為"right"，score則輸出"是右手的confidency(左手右手--classification)"。
#        #若畫面沒有手，則回傳none。

        m_time = time.time() #紀錄手勢辨識前的時間

        if results.multi_hand_landmarks:  #如果畫面有手，則會回傳一個list，裡面存有很多landmark object(landmark object裡面有x,y,z的數值，皆為-1到1之間)。若沒有手，則回傳none
            for hand_landmarks in results.multi_hand_landmarks:  #每一個hand_landmarks都是一個landmark object，裡面有x,y,z的數值。

                ##顯示那21個節點
                # -------------------------------------------------------------------------------------------------------------------------
                # -------------------------------------------------------------------------------------------------------------------------
                #引入collection，且給予別名。
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing_styles = mp.solutions.drawing_styles\
                
                #這行會使用那些collection去顯示那21個節點
                #加上手勢的那些線和節點
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                # -------------------------------------------------------------------------------------------------------------------------
                # -------------------------------------------------------------------------------------------------------------------------


                keypoint_pos = []  #keypoint_pos會存有那21個節點的x,y值(那些x,y值是以tuple去存放)

                #用hand_landmarks.landmark[i]去呼叫手掌的21個節點([0]...[20])，這21個節點都在landmark object裡面。
                #landmark object裡面有x,y,z的數值，
                # x 是用 hand_landmarks.landmark[i].x 去呼叫
                # y 是用 hand_landmarks.landmark[i].y 去呼叫
                # x,y 是節點的 x,y 值 (但那個值介於-1到1之間)
                for i in range(21):
                    x = hand_landmarks.landmark[i].x * frame.shape[1]  #frame.shape[1] 是 img 的 G-channel
                    y = hand_landmarks.landmark[i].y * frame.shape[0]  #frame.shape[0] 是 img 的 B-channel

                    #keypoint_pos會存有那21個節點的x,y值(那些x,y值是以tuple去存放)
                    #所以keypoint_pos的長度為21，型態為list。
                    keypoint_pos.append((x,y))  #會把很多型態為tuple的(x,y)都放入list。

                bounding = []

                #找出邊界
                if keypoint_pos: 
                    #print("{}       {}".format(keypoint_pos[0],keypoint_pos[4]))
                    xmin, xmax = min(keypoint_pos[:][0]), max(keypoint_pos[:][0])
                    ymin, ymax = min(keypoint_pos[:][1]), max(keypoint_pos[:][1])
                    bounding = [xmin, ymin, xmax, ymax]


                # 若影像中有手，則此時已經針對一張圖片，建立了 "存有那21個節點的x,y值" 的 keypoint_pos 這個list。
                # 若影像中沒有手，則這時keypoint_pos是空的list。

                #若keypoint_pos非空，則keypoint_pos裡面為:
                #   [2]....[4]是大拇指 / [5]....[8]是食指 / [9]....[12]是中指 / [13]...[16]是無名指 / [17]...[20]是小指

                # 只有當影像中有手，這個if裡面的程式才會執行。
                if keypoint_pos:  

                    #接下來會根據21個節點的position去算angle，再去判斷手勢。
                    # -------------------------------------------------------------------------------------------------------------------------
                    # -------------------------------------------------------------------------------------------------------------------------

                    # angle_list是一個長度為5，元素值介於0到180的list，包含5根手指個別的夾角
                    #當食指彎曲的程度比較大(角度也相對大)時，angle_list[1]的值會大於40。
                    #當食指完全伸直時，angle_list[1]的值會小於40。
                    angle_list = hand_angle(keypoint_pos)

                    # 根據角度判斷手勢是否只有伸出食指
                    # 當只有伸出食指時，回傳一個string，內有"index finger up!"。  
                    # 當手勢不是伸出食指，則回傳none。
                    gesture_str = hand_gesture(angle_list, keypoint_pos)
                    # -------------------------------------------------------------------------------------------------------------------------
                    # -------------------------------------------------------------------------------------------------------------------------

                    #根據gesture_str(也就是辨識出的手勢名稱)去做手勢相對應的處理
                    frame, gesture_str = BEHAVIOR(frame, gesture_str, keypoint_pos, angle_list, bounding)

                    str_for_put_text = gesture_str  #gesture_str的內容之後會被印在影片上

#        #用筆電前攝像頭則這段可以註解調
#        # ---------------------------------------------------------------------------           
#        h_ratio_w = frame.shape[0] / frame.shape[1]  # (高除以寬)的比
#        new_w = 400 # FIXME: 改成自己希望的顯示尺寸
#        new_h = new_w * h_ratio_w
#        frame = cv2.resize(frame, (int(new_w), int(new_h)))  #讓它等比例縮小
#        print("{}".format(frame.shape)) 
#
#
#        #調整顯示的圖片大小
#        #resize(800,540)表示將寬改成800,高改成540
#        #frame = cv2.resize(frame, (800, 540))
#
#        #印出相機原始攝像的大小
#        # pi camera v2  這個會顯示:  (864, 1536, 3)  = (h,w,c)
#        #
#        #若resize(800,540)，
#        # 則這行輸出(540,800,3)
#        #因為這行輸出的順序是(h,w,c)
#        #print("{}".format(frame.shape)) 
#
#        #圍繞x和y軸旋轉(因為樹梅派相機的影像是左右上下顛倒)
#        #筆電攝像頭則可以註解掉這行
#        frame = cv2.flip(frame, -1)  # FIXME: 按照自己的攝像頭畫面調整這個
#
#        #先翻轉畫面，再加入提示字詞，這樣提示字詞才不會被顛倒。
#        cv2.putText(frame, str_for_put_text, (10, 60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
#        # --------------------------------------------------------------------------- 

        cv2.putText(frame, str_for_put_text, (10, 60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        #顯示圖片
        cv2.imshow('MediaPipe Hands', frame) #frame此時是bgr形式，因為imshow()輸入的圖片必須是bgr形式

        f_time = time.time()
        print(50*"=")
        print("time\nbefore hand classification: {} sec \nafter hand classification: {} sec \nwhole time: {} sec".format(m_time-i_time, f_time-m_time, f_time-i_time))
        print(50*"=")

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()  #當結束執行以後，會釋出相機資源

if __name__ == '__main__':
    detect()