from collections import deque

MAX_BUF_SIZE = 144

class FrameQue:
    def __init__(self):
        self.que = deque(maxlen=MAX_BUF_SIZE)

    def put(self, frame):
        self.que.append(frame)

    def get(self):
        return self.que.popleft()

    def empty(self):
        return False if self.que else True

    def qsize(self):
        return len(self.que)