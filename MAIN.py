
from audio_recognition import audio_recognition ,isConnected
import cv2


print("欢迎使用第七代人工智能交互系统")
print("按S键开始语音识别 说\"活体检测\" or \"视觉识别\" or \"现实增强\"")
while True:
    #key = cv2.waitKey(1) & 0xFF
    # if key == ord("s"):
    if isConnected()  == False:
        print("网络链接失败，启用手动模式")
        print("输入：1 liveness demo ")
        print("输入：2 ALmighty AI CV ")
        print("输入：4 Three  dimension virtual box on chessboard ")
        print("输入：5 destroy all windows ")
        print("输入：6 wechat cloudmap ")
        print("输入：7 Find  chorus of a music ")
        result=int(input("Enter your choice"))
    else:
        result = audio_recognition()
    print("result:",result)
    if result == 1:
        result =0
        from liveness_detect import liveness_demo
        liveness_demo()
        print("terminate")
    elif result == 2:
        result = 0
        from Almighty_AI_CV import Almighty_AI_CV
        Almighty_AI_CV()
    elif result == 3:
        result =0
        print("未识别关键字 请再试一次")
        print("terminate")
    elif result == 4:
        from ThreeDbox_chessboard import drawvirtualbox
        drawvirtualbox()
        result = 0
    elif result == 5:
        cv2.destroyAllWindows()
        break
    elif result == 6:
        from py_wechat.friends import Wechat_cloudmap
        Wechat_cloudmap()
        #print("该功能未适配MacOS 敬请期待！")
        result=0
    elif result == 7:
        from pychorusmaster.pychorus_main import Find_chorus
        Find_chorus()
        print("inspection complete!")
    result="END"



