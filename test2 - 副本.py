import base64
import json
import re
import sys
import time
import wave

import pyaudio


def audio_recognize():
    # 定义数据流块
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    # 录音时间
    RECORD_SECONDS = 5
    # 要写入的文件名
    WAVE_OUTPUT_FILENAME = "output.wav"
    # 创建PyAudio对象
    p = pyaudio.PyAudio()

    # 打开数据流
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    # 开始录音
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    # 停止数据流
    stream.stop_stream()
    stream.close()

    # 关闭PyAudio
    p.terminate()

    # 写入录音文件
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    IS_PY3 = sys.version_info.major == 3

    if IS_PY3:
        from urllib.request import urlopen
        from urllib.request import Request
        from urllib.error import URLError
        from urllib.parse import urlencode
        timer = time.perf_counter
    else:
        from urllib2 import urlopen
        from urllib2 import Request
        from urllib2 import URLError
        from urllib import urlencode
        if sys.platform == "win32":
            timer = time.clock
        else:
            # On most other platforms the best timer is time.time()
            timer = time.time

    API_KEY = '1XktiGYNDDjimHk8iWmpyVE1'
    SECRET_KEY = 'YcO0tbZFoN98lx0mH5PfL3KlZ06GdLFc'

    # 需要识别的文件
    AUDIO_FILE = 'output.wav'  # 只支持 pcm/wav/amr 格式，极速版额外支持m4a 格式
    # 文件格式
    FORMAT = AUDIO_FILE[-3:]  # 文件后缀只支持 pcm/wav/amr 格式，极速版额外支持m4a 格式

    CUID = '123456PYTHON'
    # 采样率
    RATE = 16000  # 固定值

    # 普通版

    DEV_PID = 1536  # 1537 表示识别普通话，使用输入法模型。1536表示识别普通话，使用搜索模型。根据文档填写PID，选择语言及识别模型
    ASR_URL = 'http://vop.baidu.com/server_api'
    SCOPE = 'audio_voice_assistant_get'  # 有此scope表示有asr能力，没有请在网页里勾选，非常旧的应用可能没有

    # 测试自训练平台需要打开以下信息， 自训练平台模型上线后，您会看见 第二步：“”获取专属模型参数pid:8001，modelid:1234”，按照这个信息获取 dev_pid=8001，lm_id=1234
    # DEV_PID = 8001 ;
    # LM_ID = 1234 ;

    # 极速版 打开注释的话请填写自己申请的appkey appSecret ，并在网页中开通极速版（开通后可能会收费）

    # DEV_PID = 80001
    # ASR_URL = 'http://vop.baidu.com/pro_api'
    # SCOPE = 'brain_enhanced_asr'  # 有此scope表示有极速版能力，没有请在网页里开通极速版

    # 忽略scope检查，非常旧的应用可能没有
    # SCOPE = False

    class DemoError(Exception):
        pass

    """  TOKEN start """

    TOKEN_URL = 'http://openapi.baidu.com/oauth/2.0/token'

    def fetch_token():
        params = {'grant_type': 'client_credentials',
                  'client_id': API_KEY,
                  'client_secret': SECRET_KEY}
        post_data = urlencode(params)
        if (IS_PY3):
            post_data = post_data.encode('utf-8')
        req = Request(TOKEN_URL, post_data)
        try:
            f = urlopen(req)
            result_str = f.read()
        except URLError as err:
            print('token http response http code : ' + str(err.code))
            result_str = err.read()
        if (IS_PY3):
            result_str = result_str.decode()

        # print(result_str)
        result = json.loads(result_str)
        # print(result)
        if ('access_token' in result.keys() and 'scope' in result.keys()):
            print(SCOPE)
            if SCOPE and (not SCOPE in result['scope'].split(' ')):  # SCOPE = False 忽略检查
                raise DemoError('scope is not correct')
            print('SUCCESS WITH TOKEN: %s  EXPIRES IN SECONDS: %s' % (result['access_token'], result['expires_in']))
            return result['access_token']
        else:
            raise DemoError(
                'MAYBE API_KEY or SECRET_KEY not correct: access_token or scope not found in token response')

    """  TOKEN end """

    if __name__ == '__main__':
        token = fetch_token()

        speech_data = []
        with open(AUDIO_FILE, 'rb') as speech_file:
            speech_data = speech_file.read()

        length = len(speech_data)
        if length == 0:
            raise DemoError('file %s length read 0 bytes' % AUDIO_FILE)
        speech = base64.b64encode(speech_data)
        if (IS_PY3):
            speech = str(speech, 'utf-8')
        params = {'dev_pid': DEV_PID,
                  # "lm_id" : LM_ID,    #测试自训练平台开启此项
                  'format': FORMAT,
                  'rate': RATE,
                  'token': token,
                  'cuid': CUID,
                  'channel': 1,
                  'speech': speech,
                  'len': length
                  }
        post_data = json.dumps(params, sort_keys=False)
        # print post_data
        req = Request(ASR_URL, post_data.encode('utf-8'))
        req.add_header('Content-Type', 'application/json')
        try:
            begin = timer()
            f = urlopen(req)
            result_str = f.read()
            print("Request time cost %f" % (timer() - begin))
        except URLError as err:
            print('asr http response http code : ' + str(err.code))
            result_str = err.read()

        if (IS_PY3):
            result_str = str(result_str, 'utf-8')
        result = re.findall('\"result\"\:\[\"(\w+)\"\]', result_str, re.S)
        print("Ending-------------------------------------------------")
        # print(result_str)
        print(result)
        liveness_detect = re.search('活体检测', result_str)
        cv_recognize = re.search('视觉识别', result_str)
        with open("result.txt", "w") as of:
            of.write(result_str)
        if liveness_detect:
            return 1
        elif cv_recognize:
            return 2
        else:
            return 3













































