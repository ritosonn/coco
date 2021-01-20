from typing import NamedTuple
import yaml
from typing import List
import numpy as np

class Rect(NamedTuple):
    l: float
    t: float
    w: float
    h: float
    cls: int

class Encoder:
    def __init__(self, h: int, w: int, s: int):
        self.h = h
        self.w = w
        self.s = s
        self.cls = np.zeros((h, w), dtype=np.uint8)
        self.reg = np.zeros((h, w, 4), dtype=np.float32)
        self.center = np.zeros((h, w), dtype=np.float32)
    
    def encode(self, r: Rect):
        for y in range(self.h):
            y_emb = self.s // 2 + y * self.s
            if not (r.t <= y_emb <= r.t + r.h):
                continue
            for x in range(self.w):
                x_emb = self.s // 2 + x * self.s
                if not (r.l <= x_emb <= r.l + r.w):
                    continue
                self.cls[y, x] = r.cls
                l_ = x_emb - r.l
                t_ = y_emb - r.t
                r_ = r.l + r.w - x_emb
                b_ = r.t + r.h - y_emb
                self.reg[y, x] = [l_, t_, r_, b_]
                self.center[y, x] = (min(l_, r_)/max(l_, r_) * min(t_, b_)/max(t_, b_)) ** 0.5

    def decode(self):
        v: List[Rect] = []
        for y in range(self.h):
            y_emb = self.s // 2 + y * self.s
            for x in range(self.w):
                if self.cls[y, x] > 0:
                    x_emb = self.s // 2 + x * self.s
                    l_, t_, r_, b_ = self.reg[y, x]
                    v.append(Rect(x_emb - l_, y_emb - t_, l_ + r_, t_ + b_, self.cls[y, x]))
        return v


def encode_decode_check():
    np.set_printoptions(threshold=10000)
    r1 = Rect(107, 203, 71, 82, 1)
    r2 = Rect(301, 205, 53, 64, 2)
    encoder = Encoder(10, 10, 40)
    encoder.encode(r1)
    encoder.encode(r2)
    print(encoder.cls, encoder.reg, encoder.center)
    v = encoder.decode()
    print(v)


def main():
    with open("/dataset/coco2017/annotations/instances_val2017.json") as f:
        data = yaml.safe_load(f)
    print(data["info"])

if __name__ == "__main__":
    encode_decode_check()