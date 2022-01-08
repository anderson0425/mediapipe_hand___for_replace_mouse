import pyautogui

#pyautogui.size()回傳螢幕尺寸
screenWidth, screenHeight = pyautogui.size() 
print("screenWidth: {}   screenHeight: {}".format(screenWidth, screenHeight))

#pyautogui.position()回傳滑鼠游標位置。
currentMouseX, currentMouseY = pyautogui.position()
print("currentMouseX: {}   currentMouseY: {}".format(currentMouseX, currentMouseY))

#把滑鼠游標移到(x=100, y=150)的位置
pyautogui.moveTo(100, 150)

#用2秒的時間，完成把滑鼠游標移到(x=100, y=150)的位置這個動作
pyautogui.moveTo(500, 500, duration=2, tween=pyautogui.easeInOutQuad)

#讓滑鼠做出點擊的動作
pyautogui.click()

#讓滑鼠在(x=200, y=220)的位置做出點擊的動作
pyautogui.click(200, 220)

#從當前位置，讓滑鼠游標做出(x=+0, y=+10)的移動。
#由於越往下，y值越大，因此這會使滑鼠垂直向下移動10單位pixel。
pyautogui.move(0, 10) 

# move the mouse down 50 pixels.
pyautogui.move(0, 50)       

# move the mouse left 30 pixels.
 pyautogui.move(-30, 0)      

#讓滑鼠做出雙重點擊的動作
pyautogui.doubleClick()

#回傳true/false，可檢驗(x=1000,y=2000)是否超出這台電腦螢幕的長寬。
pyautogui.onScreen(1000, 2000)

# drag mouse to X of 100, Y of 200 while holding down left mouse button
#在2秒的時間內，做出:  moveto(100, 200) + 一直按著滑鼠左鍵
#會有"框選"的效果
#但滑鼠會移動過頭，而且2秒後，那個左鍵就會release，也就沒有框選的效果了。
pyautogui.dragTo(100, 200, duration=2,  button='left')    

# pyautogui.easeInQuad:  start slow, end fast
pyautogui.moveTo(100, 100, 2, pyautogui.easeInQuad)   

# pyautogui.easeOutQuad: start fast, end slow
pyautogui.moveTo(100, 100, 2, pyautogui.easeOutQuad)  

# pyautogui.easeInOutQuad: start and end fast, slow in middle
pyautogui.moveTo(100, 100, 2, pyautogui.easeInOutQuad)  

# pyautogui.easeInBounce: bounce at the end
#看不懂這個，少用ㄅ
pyautogui.moveTo(100, 100, 2, pyautogui.easeInBounce)   

# pyautogui.easeInElastic: rubber band at the end
#看不懂這個，少用ㄅ
pyautogui.moveTo(100, 100, 2, pyautogui.easeInElastic)  

#每個char跟char之間間隔0.001秒。
pyautogui.write('Hello world!', interval=0.001)

#會按壓按鍵'esc'，然後釋放它
#因此press('esc')其實是keyDown('esc') + keyUp('esc')
pyautogui.press('esc') # Simulate pressing the Escape key.

#keyDown('ctrl')會一直按壓ctrl，不釋放。
#理想來說應該要是一直按壓，但他似乎有時間限制，時間一到，程式執行結束，就放開了。
pyautogui.keyDown('ctrl')  # hold down the ctrl key

#keyUp('shift')會(release)釋放ctrl按鍵。
pyautogui.keyUp('shift')

#會執行6次press滑鼠左鍵
#效果跟這個相同
#pyautogui.press(['left', 'left', 'left', 'left', 'left', 'left'])
#也跟這個相同
#pyautogui.press('left', presses=6)
pyautogui.write(['left', 'left', 'left', 'left', 'left', 'left'])

#the hotkey() can be passed several key strings 
# which will be pressed down in order, and then released in reverse order. 
#會先按壓ctrl、再按壓c，這些都按住不放，
#接著會先release 'c'、再release 'ctrl'
pyautogui.hotkey('ctrl', 'c')

#the hotkey() can be passed several key strings 
# which will be pressed down in order, and then released in reverse order. 
#會先按壓ctrl、再按壓shift、再按壓esc，這些都按住不放，
#接著會先release 'esc'、再release 'shift'、再release 'ctrl'
pyautogui.hotkey('ctrl', 'shift', 'esc')