# mediapipe_hand___for_replace_mouse
1.只伸出食指中指，且食指中指夾緊，則可以控制滑鼠游標

2.只伸出食指，則可以做出滑鼠左鍵點擊一次的效果(有做delay的設定，不會因太敏感而重複單擊)

3.只伸出無名指和小指，可以做出滑鼠左鍵點擊2次的效果(有做delay的設定，不會因太敏感而連續雙擊)

4.只伸出大拇指食指，也食指朝左，則無動作，這個可以自己加上功能(在line 772)

5.只伸出大拇指食指中指，則螢幕截圖，儲存在[本機]-->[圖片]-->[螢幕擷取畫面]

6.只伸出小指，則可以做出滑鼠拖曳(drag)的效果。



還可以做調整的幾個方向:

1.[角度改良]
針對手勢移動時，手勢的角度造成的手勢誤判做出修正，希望不同角度也可以做出正確手勢判斷

2.[手勢改良]
測試哪個"手勢組"比較不會誤判，或讓誤判發生時造成的error不會有太大影響。

3.[影像改良]
測試如何處理影像可以兼顧讓影像清晰可以手勢辨識，但是又可以去除背景的光害(讓手指可以不受光害影響正確辨識)

4.[遊戲化]
將其套用在pygame做出一個簡易遊戲
