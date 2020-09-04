from audio_recognition import audio_recognition
import cv2
import threading
import time



print("欢迎使用第七代人工智能交互系统")
print("按S键开始语音识别 说\"活体检测\" or \"视觉识别\" or \"现实增强\"")
""""
class Audio_recognition(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        from audio_recognition import audio_recognition
        self.result= audio_recognition()
        print("result2:self.result")
    def get_result(self):
        return self.result
"""
class liveness_demo(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.__flag=threading.Event()
        self.__flag.set()
        self.__running=threading.Event()
        self.__running.set()
    def run(self):
        #threadLock.acquire()
        if self.__running.is_set():
            self.__flag.wait()
            from liveness_detect import liveness_demo
            liveness_demo()

        #threadLock.release()
    def pause(self):
        self.__flag.clear()
    def resume(self):
        self.__flag.set()
    def stop(self):
        self.__flag.set()
        self.__running.clear()
class Almighty_AI_CS(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        from Almighty_AI_CV import Almighty_AI_CS
        Almighty_AI_CS()

class ThreeDbox_chessboard(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        from ThreeDbox_chessboard import drawvirtualbox
        drawvirtualbox()

#Audio_recognition_Thread=Audio_recognition()
liveness_demo_Thread=liveness_demo()
Almighty_AI_CS_Thread=Almighty_AI_CS()
ThreeDbox_chessboard_Thread=ThreeDbox_chessboard()
threadLock=threading.Lock()

liveness_demo_Thread_status=0
while True:
    #key = cv2.waitKey(1) & 0xFF
    # if key == ord("s"):
    #threads=[]
    #Audio_recognition_Thread.start()

    #threads.append(Audio_recognition_Thread)
    #for t in threads:
    #    print("6666666666666666",t)
    #    t.join()
    #result = Audio_recognition_Thread.get_result()
    #Audio_recognition_Thread.join()
    result = audio_recognition()
    print("result:----------------------------",result)
    if result == 1:
        result =0
        if liveness_demo_Thread_status==0:
            liveness_demo_Thread.start()
            liveness_demo_Thread_status=1
        liveness_demo_Thread.resume()
        print("terminate")
    elif result == 2:
        result = 0
        Almighty_AI_CS_Thread.start()
    elif result == 3:
        result =0
        print("未识别关键字 请再试一次")

    elif result == 4:
        result = 0
        ThreeDbox_chessboard_Thread.start()
    elif result == 5:
        liveness_demo_Thread.pause()
        liveness_demo_Thread.stop()
        #Almighty_AI_CS_Thread.join()
        #ThreeDbox_chessboard_Thread.join()
        print("ALL ABORT!")
        break
    result="END"
    print("Done!!!!!!!!!!!!!!!!!")





