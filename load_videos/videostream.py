import queue
import threading
from datetime import datetime
import time

import cv2


class QueuedStream:

    def __init__(self, uri, drop=True):
        self.uri = uri
        self.queue = queue.Queue(maxsize=1)
        self.lock_started = threading.Lock()
        self.fps = 25
        self.opened = False
        self.stopped = False
        self.drop = drop
        self.start_time = 0
        self.sleep_between_frame = True

    def start(self):
        self.lock_started.acquire()
        self.th = threading.Thread(target=self._thread_func)
        self.th.daemon = True
        self.th.start()
        self.lock_started.acquire()

    def read(self):
        if not self.stopped:
            frame, frame_id, time_stamp = self.queue.get(True)
            if frame is None:
                return (False, None, None, None)
            return (True, frame, frame_id, time_stamp)
        else:
            return (False, None, None, None)

    def stop(self):
        if not self.stopped:
            self.stopped = True
            try:
                self.queue.get(False)
            except Exception:
                pass
            self.th.join()

    def isOpened(self):
        return self.opened

    def release(self):
        self.stop()

    def estimate_framerate(self):
        return self.fps

    def set_start_time(self, ms):
        self.start_time = ms

    def set_sleep_between_frame(self, v):
        self.sleep_between_frame = v

    def _thread_func(self):
        '''keep looping infinitely'''
        global IMAGE_W, IMAGE_H, SCALE_RATIO
        if len(self.uri) == 0:
            stream = cv2.VideoCapture(0)
            estimate_fps = True
            get_time_stamp = True
        else:
            stream = cv2.VideoCapture(self.uri)
            if self.uri.startswith('rtsp://'):
                estimate_fps = True
                get_time_stamp = True
            else:
                estimate_fps = False
                get_time_stamp = False
                self.fps = stream.get(5)
        if self.start_time != 0:
            stream.set(0, self.start_time)
        time.sleep(0.1)
        self.opened = stream.isOpened()

        self.lock_started.release()
        if not self.opened:
            stream.release()
            return

        start_time = time.time()
        frame_id = 0

        while not self.stopped:

            grabbed, frame = stream.read()

            if not grabbed:
                frame = None
                frame_id = None
            else:
                frame_id += 1

            if get_time_stamp:
                time_stamp = datetime.now()
            else:
                time_stamp = None

            if self.drop:
                try:
                    self.queue.get(False)
                except Exception:
                    pass
                self.queue.put((frame, frame_id, time_stamp))
            else:  # not drop
                self.queue.put((frame, frame_id, time_stamp))

            if frame is None:
                stream.release()
                return

            if not estimate_fps:
                if self.sleep_between_frame:
                    time.sleep(1.0 / self.fps)
            else:
                if frame_id > 25 and self.drop:
                    self.fps = frame_id / (time.time() - start_time)
                elif frame_id > 5 and self.drop:
                    estimate = frame_id / (time.time() - start_time)
                    self.fps = (self.fps + estimate) / 2.0
        # stopped
        if frame is not None:  # stopped.value == True
            try:
                self.queue.get(True, 0.5)
            except Exception:
                pass
        else:
            self.stopped = True
        stream.release()


#if __name__ == '__main__':
#
#    video_path = r"C:\Users\Thanh\Downloads\2019-11-24_14-00-00_cam_1.mp4"
#    stream = QueuedStream(video_path)
#    stream.start()
#
#    while True:
#        ret, frame, frame_id, time_stamp = stream.read()
#        if time_stamp:
#            print("{}".format(time_stamp.strftime("%m/%d/%Y, %H:%M:%S")))
#        if ret is False:
#            print('Videos is ended')
#            # self.run = False
#            break
#
#        cv2.imshow("", frame)
#        cv2.waitKey(1)
