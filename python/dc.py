# -*- coding: utf-8 -*-
# dc.py

import math
import numpy as np

# constant
BLACK = 0
WHITE = 1

N_ENDS = 10 # num of ends
END_FIRST = 0
END_LAST = N_ENDS - 1

N_TURNS = 16 # num of turns
TURN_FIRST = 0
TURN_LAST = N_TURNS - 1

def to_turn_color(t):
    return t % 2

N_COLOR_STONES = 8
N_STONES = N_COLOR_STONES * 2

SCORE_MIN = -N_COLOR_STONES
SCORE_MAX = N_COLOR_STONES
SCORE_LENGTH = SCORE_MAX - SCORE_MIN + 1

# physics constant
STONE_RADIUS = 0.145
HOUSE_RADIUS = 1.83

W_SPIN = 0.066696

PLAYAREA_WIDTH = 4.75
PLAYAREA_LENGTH = 8.23

X_TEE = PLAYAREA_WIDTH / 2
Y_TEE = 3.05 + HOUSE_RADIUS

X_PLAYAREA_MIN = X_TEE - PLAYAREA_WIDTH / 2
X_PLAYAREA_MAX = X_TEE + PLAYAREA_WIDTH / 2
Y_PLAYAREA_MIN = Y_TEE + HOUSE_RADIUS - PLAYAREA_LENGTH
Y_PLAYAREA_MAX = Y_TEE + HOUSE_RADIUS

X_THROW = X_TEE
Y_THROW = Y_PLAYAREA_MIN + 30.0

R_IN_HOUSE = HOUSE_RADIUS + STONE_RADIUS
R2_IN_HOUSE = R_IN_HOUSE ** 2

XY_TEE = (X_TEE, Y_TEE)
XY_THROW = (X_THROW, Y_THROW)

VX_TEE_SHOT_R = -0.99073974
VY_TEE_SHOT = -29.559775

RIGHT = 0
LEFT = 1

TEE_SHOT_R = (VX_TEE_SHOT_R, VY_TEE_SHOT, RIGHT)

ERROR_SIGMA = 0.145
ERROR_SCALE_X = 0.5 # gat version
ERROR_SCALE_Y = 2.0 # gat version
    
VX_ERROR_SIGMA = 0.117659 * ERROR_SCALE_X
VY_ERROR_SIGMA = 0.0590006 * ERROR_SCALE_Y

def is_in_house_r(r):
    return bool(r < R_IN_HOUSE)

def is_in_house_r2(r2):
    return bool(r2 < R2_IN_HOUSE)

def is_in_house_xy(x, y):
    dx = x - X_TEE
    dy = y - Y_TEE
    return is_in_house_r2(dx * dx + dy * dy)

def is_in_house(pos):
    return is_in_house_xy(pos[0], pos[1])

def is_in_play_area_xy(x, y):
    return bool((X_PLAYAREA_MIN < x) and (x < X_PLAYAREA_MAX) and (Y_PLAYAREA_MIN < y) and (y < Y_PLAYAREA_MAX))

def is_in_play_area(pos):
    return is_in_play_area_xy(pos[0], pos[1])

def calc_r2(a, b):
    return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2
def calc_r(a, b):
    return np.hypot(b[0] - a[0], b[1] - a[1])
def calc_th(a, b = None):
    if b is None:
        return np.arctan2(a[0], a[1])
    else:
        return np.arctan2(b[0] - a[0], b[1] - a[1])

def calc_v2(vxy):
    return (vxy[0] ** 2) + (vxy[1] ** 2)

def calc_v(vxy):
    return np.hypot(vxy[0], vxy[1])

class Board:
    def __init__(self):
        self.init()
    
    def init(self):
        self.end = END_FIRST
        self.turn = TURN_FIRST
        self.score = np.zeros((2), dtype = int)
        self.stone = np.empty(N_STONES, dtype = tuple)
        self.locate_in_throw_point()
    
    def locate_in_throw_point(self):
        for i in range(N_STONES):
            self.stone[i] = XY_THROW

def count_in_house_a(sa, color = None): # count num of stones in house
    cnt = 0
    if color is None:
        lst = range(N_STONES)
    else:
        lst = range(color, N_STONES, 2)
    for i in lst:
        if is_in_house(sa[i]):
            cnt += 1
    return cnt

def count_in_play_area_a(sa, color = None): # count num of stones in play area
    cnt = 0
    if color is None:
        lst = range(N_STONES)
    else:
        lst = range(color, N_STONES, 2)
    for i in lst:
        if is_in_play_area(sa[i]):
            cnt += 1
    return cnt

def count_score_a(sa): # count stone score by array
    bmin2 = R2_IN_HOUSE
    wmin2 = R2_IN_HOUSE
    for i in range(BLACK, N_STONES, 2):
        st = sa[i]
        if is_in_play_area(st):
            r2 = calc_r2(st, XY_TEE)
            bmin2 = min(bmin2, r2)
    for i in range(WHITE, N_STONES, 2):
        st = sa[i]
        if is_in_play_area(st):
            r2 = calc_r2(st, XY_TEE)
            wmin2 = min(wmin2, r2)
    cnt = 0
    if bmin2 > wmin2:
        for i in range(WHITE, N_STONES, 2):
            st = sa[i]
            if is_in_play_area(st):
                r2 = calc_r2(st, XY_TEE)
                if r2 < bmin2:
                    cnt -= 1
    elif bmin2 < wmin2:
        for i in range(BLACK, N_STONES, 2):
            st = sa[i]
            if is_in_play_area(st):
                r2 = calc_r2(st, XY_TEE)
                if r2 < wmin2:
                    cnt += 1
    return cnt

def count_score(bd): # count stone score on a board
    return count_score_a(bd.stone)

def is_caving_in_pp(p0, p1):
    return (calc_r2(p0, p1) < ((2 * STONE_RADIUS) ** 2))

def is_caving_in_bp(bd, p):
    for i in range(N_STONES):
        if is_caving_in_pp(bd.stone[i], p):
            return True
    return False

def locate_in_play_area_p():
    # locate to random position in the play area
    return (X_PLAYAREA_MIN + np.random.rand() * PLAYAREA_WIDTH,
            Y_PLAYAREA_MIN + np.random.rand() * PLAYAREA_LENGTH)

def locate_in_house_p():
    # locate to random position in the house
    r = np.random.rand() * R_IN_HOUSE
    th = np.random.rand() * 2 * math.pi
    return (X_TEE + r * math.sin(th), Y_TEE + r * math.cos(th))

def locate_in_play_area_b(nb, nw):
    # locate to random positions in the play area
    bd = Board()
    bd.locate_in_throw_point()
    for i in range(nb): # black
        while True:
            pos = locate_in_play_area_p()
            if not is_caving_in_bp(bd, pos): # ok
                bd.stone[i * 2] = pos
                break
    for i in range(nw): # white
        while True:
            pos = locate_in_play_area_p()
            if not is_caving_in_bp(bd, pos): # ok
                bd.stone[i * 2 + 1] = pos
                break
    return bd