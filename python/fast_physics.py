# -*- coding: utf-8 -*-
# dc_simulation.py

import math, copy
import numpy as np

import dc

# simulation constant
FRICTION_RINK = 12.009216 # ( = g * mu )
FRICTION_STONES = FRICTION_RINK # strange setting in official rule

DX_V_R = 0.001386965639075
DY_V_R = 0.041588442394742

#DR_V_R = np.hypot(DX_V_R, DY_V_R)
DR_V_R = 0.041611563471

PHI = math.atan2(DY_V_R, DX_V_R)

ALPHA = 0.03333564

#B = (math.cos(ALPHA) ** 2) / (1 - (math.cos(ALPHA) ** 2))
B = 29.986811440344486

#A = DX_V_R * (dc.calc_r(dc.XY_TEE, dc.XY_THROW) / DR_V_R) / (math.cos(PHI) * math.exp(SPIRAL_B * PHI))
A = 3.45628911574e-19

R_COLLISION = 0.000004
DT_COLLISION = 0.000001

def rotationMatrix(rad):
    return np.array([[math.cos(rad), -math.sin(rad)],
                     [math.sin(rad), math.cos(rad)]])

MAT_ALPHA = [rotationMatrix(-ALPHA), rotationMatrix(ALPHA)]

# stepwise function
def calc_next_vtheta(vxy, w):
    v = dc.calc_v(vxy)
    normalized_vxy = vxy / v
    if w > 0:
        angle = -math.pi / 2
    else:
        angle = math.pi / 2
    dvxy = np.matmul(rotationMatrix(angle), normalized_vxy) * FRICTION_RINK
    return (vxy[0] + dvxy[0], vxy[1] + dvxy[1]), w

# curve function
def calc_r_by_theta(theta):
    return A * math.exp(B * theta)

def calc_theta_by_r(r):
    #print(r)
    return math.log(r / A) / B

def calc_x_by_r_theta(r, theta):
    return r * math.cos(theta)
def calc_y_by_r_theta(r, theta):
    return r * math.sin(theta)

def calc_x_by_theta(theta):
    return calc_r_by_theta(theta) * math.cos(theta)
def calc_y_by_theta(theta):
    return calc_r_by_theta(theta) * math.sin(theta)

def calc_r_by_v(v):
    return DR_V_R * v * v

def calc_v_by_r(r):
    return math.sqrt(r / DR_V_R)

def calc_v_by_theta(theta):
    return calc_v_by_r(calc_r_by_theta(theta))

def calc_theta_by_v(v):
    return calc_theta_by_r(calc_r_by_v(v))

def calc_dxy_by_vxys(vxy, spin):
    return DR_V_R * dc.calc_v(vxy) * np.matmul(MAT_ALPHA[spin], vxy)
    #return np.matmul(MAT_ALPHA[spin], [(vxy[0] ** 2) * DX_V_R, (vxy[1] ** 2) * DY_V_R])

def calc_xy_by_xy_vxyw(oxy, vxy, w):
    dxy = calc_dxy_by_vxys(vxy, int(w < 0))
    #print("dxy = ", dxy)
    dx = dxy[0]
    #if w < 0:
    #dx = -dx
    return (oxy[0] + dx, oxy[1] + dxy[1])

def calc_xy_by_vxyw_from_throw(vxy, w):
    return calc_xy_by_xy_vxyw(dc.XY_THROW, vxy, w)

def calc_v_by_t(t):
    return t * FRICTION_RINK

def calc_t_by_v(v):
    return v / FRICTION_RINK

def calc_vtheta_by_theta(theta, w = 1):
    return theta + ALPHA if w < 0 else theta - ALPHA

def calc_vtheta_by_v(v):
    return calc_vtheta_by_theta(calc_theta_by_v(v))

def calc_vxy_by_theta(theta):
    v = calc_v_by_theta(theta)
    vtheta = calc_vtheta_by_theta(theta)
    return (v * math.cos(vtheta), v * math.sin(vtheta))
    #return np.array([v * math.cos(vtheta), v * math.sin(vtheta)], dtype = np.float32)

def calc_next_theta_by_theta(theta):
    return theta

def calc_vxy_by_xyw(oxy, xy, w):
    v = calc_v_by_r(dc.calc_r(oxy, xy))
    #print("theta =", dc.calc_th(oxy, xy))
    theta = dc.calc_th(oxy, xy)
    #print(theta)
    vtheta = calc_vtheta_by_theta(theta, w)
    vx = v * math.sin(vtheta)
    return (vx, v * math.cos(vtheta))

def calc_vxy_by_xyw_from_throw(xy, w):
    return calc_vxy_by_xyw(dc.XY_THROW, xy, w)

def calc_r_lb_by_lv(l, v):
    return calc_r_by_v(v) - l

def calc_r_ub_by_lv(l, v):
    R = calc_r_by_v(v)
    return math.sqrt((R ** 2) + (l ** 2) - 2 * R * l * math.cos(ALPHA))

def calc_r_by_lv(l, v):
    l_line = l + R_COLLISION
    r = 0
    theta = 0
    r_org = calc_r_by_v(v)
    theta_org = calc_theta_by_r(r_org)

    l_tmp = r_org
    steps = 0
    
    #print(0, l_tmp - l)
    
    while l_tmp > l_line:
        r += l_tmp - l
        theta = calc_theta_by_r(r)
        
        l_tmp = math.sqrt((r ** 2) + (r_org ** 2) - 2 * r * r_org * math.cos(theta_org - theta))

        steps += 1
    
        #print(steps, l_tmp - l)
    return r

def calc_vxy_by_xylvw(oxy, xy, v, w):
    l = dc.calc_r(oxy, xy)
    R = calc_r_by_v(v)
    r = calc_r_by_lv(l, v)

    cos_alpha = ((l ** 2) + (R ** 2) - (r ** 2)) / (2 * l * R)
    alpha = math.acos(cos_alpha)
    #print("alpha = ", alpha)
    #print(oxy, xy)
    #print(dc.calc_th(oxy, xy))
    if w < 0:
        alpha = -alpha
    vtheta = calc_vtheta_by_theta(dc.calc_th(oxy, xy) + alpha, w)
    #print("vtheta = ", vtheta)
    return np.matmul(rotationMatrix(-vtheta), (0, v))

def calc_vxy_by_xylvw_from_throw(xy, v, w):
    return calc_vxy_by_xylvw(dc.XY_THROW, xy, v, w)

def calc_vxy_by_xylrw(oxy, xy, _R, w):
    l = dc.calc_r(oxy, xy)
    R = l + _R # not acculate
    v = calc_v_by_r(R)
    r = calc_r_by_lv(l, v)
    
    cos_alpha = ((l ** 2) + (R ** 2) - (r ** 2)) / (2 * l * R)
    alpha = math.acos(cos_alpha)
    if w < 0:
        alpha = -alpha
    vtheta = calc_vtheta_by_theta(dc.calc_th(oxy, xy) + alpha, w)
    return np.matmul(rotationMatrix(-vtheta), (0, v))

def calc_vxy_by_xylrw_from_throw(xy, p, w):
    return calc_vxy_by_xylrw(dc.XY_THROW, xy, p, w)

def step_xy_vxy_by_xy_vxywt(xy, vxy, w, t):
    V = dc.calc_v(vxy)
    R = calc_r_by_v(V)
    Theta = calc_theta_by_r(R)
    v = V - FRICTION_RINK * t
    r = calc_r_by_v(v)
    #print(V, v, r)
    theta = calc_theta_by_r(r)
    print(r, Theta - theta)
    l2 = (R ** 2) + (r ** 2) - 2 * R * r * math.cos(Theta - theta)
    l = math.sqrt(l2)
    #print("l =", l)
    #cos_beta = ((l ** 2) + (R ** 2) - (r ** 2)) / (2 * l * R)
    #beta = math.acos(cos_beta)
    sin_beta = r / l * math.sin(Theta - theta)
    beta = math.asin(sin_beta)
    #print("beta =", beta)
    
    dtheta = Theta - theta
    alpha = ALPHA

    if w < 0:
        dtheta = -dtheta
        alpha = -alpha
        beta = -beta
    
    #print("ovtheta =", dc.calc_th(vxy), "dtheta =", dtheta)

    ntheta = dc.calc_th(vxy) + alpha - beta
    nvtheta = dc.calc_th(vxy) + dtheta

    #print("ntheta =", ntheta, "nvtheta =", nvtheta)

    nxy = (xy[0] + l * math.sin(ntheta), xy[1] + l * math.cos(ntheta))

    nvxy = (v * math.sin(nvtheta), v * math.cos(nvtheta))

    return nxy, nvxy

# move limit function
def calc_nr_by_aw_asl(xy, vxy, w, gxy, asxy):
    r = dc.calc_r(xy, asxy) - 2 * dc.STONE_RADIUS
    gr = dc.calc_r(xy, gxy)
    if gr < r:
        return 0
    nr2 = (gr ** 2) + (r ** 2) - 2 * gr * r * math.cos(ALPHA)
    return math.sqrt(nr2)
    
def calc_nv_by_aw_asl(xy, vxy, w, gxy, asxy):
    return calc_v_by_r(calc_nr_by_aw_asl(xy, vxy, w, gxy, asxy))
    
def calc_nt_by_aw_asl(xy, vxy, w, gxy, asxy):
    return calc_t_by_v(calc_nv_by_aw_asl(xy, vxy, w, gxy, asxy))

def calc_dt_by_aw_asl(xy, vxy, w, gxy, asxy):
    gt = calc_t_by_v(dc.calc_v(vxy))
    return gt - calc_nt_by_aw_asl(xy, vxy, w, gxy, asxy)

def calc_dt_by_aw_aw(xy0, vxy0, w0, gxy0, xy1, vxy1, w1, gxy1):
    r = dc.calc_r(xy0, xy1) - 2 * dc.STONE_RADIUS
    return r / (dc.calc_v(vxy0) + dc.calc_v(vxy1))

# collision
def is_collisious_aw_aw(xy0, vxy0, xy1, vxy1):
    return (xy1[0] - xy0[0]) + (vxy1[0] - vxy0[0]) + (xy1[1] - xy0[1]) * (vxy1[1] - vxy0[1]) < 0

def is_collisious_aw_as(xy0, vxy0, xy1):
    return True

def collision_aw_asl(xy, vxy, w, asxy):
    th_col = dc.calc_th(xy, asxy)
    v, th = dc.calc_v(vxy), dc.calc_th(vxy)
    # relative velocity(+)
    vt = v * math.sin(th - th_col) # tangent velocity
    vn = max(0.0, v * math.cos(th - th_col)) # normal velocity
    
    v_threshold = 1
    ni_m = vn
    if vn <= v_threshold:
        ni_m *= 0.5
    maxFriction = FRICTION_STONES * ni_m
    oldTangentLambda = vt / 6.0
    ti_m = max(-maxFriction, min(oldTangentLambda, maxFriction))
    dw = ti_m * 2.0 / dc.STONE_RADIUS
    
    # update
    asv = np.hypot(ni_m, ti_m)
    asth = np.arctan2(ni_m, ti_m) + th_col
    asw = dw
    asvxy = (asv * math.sin(asth), asv * math.cos(asth))
                
    awv = np.hypot(vn - ni_m, vt - ti_m)
    awth = np.arctan2(vn - ni_m, vt - ti_m) + th_col
    aww = w + dw
    awvxy = (awv * math.sin(awth), awv * math.cos(awth))
                
    return awvxy, aww, asvxy, asw

def collision_aw_aw(xy0, vxy0, w0, xy1, vxy1, w1):
    th_col = dc.calc_th(xy0, xy1)
    
    v0, th0 = dc.calc_v(vxy0), dc.calc_th(vxy0)
    v0n = v0 * math.cos(th0 - th_col)
    v0t = v0 * math.sin(th0 - th_col)
    
    v1, th1 = dc.calc_v(vxy1), dc.calc_th(vxy1)
    v1n = v1 * math.cos(th1 - th_col)
    v1t = v1 * math.sin(th1 - th_col)
            
    vn, vt = v0n - v1n, v0t - v1t
        
    v_threshold = 1
    ni_m = vn
    if abs(vn) <= v_threshold:
        ni_m *= 0.5
    maxFriction = FRICTION_STONES * ni_m
    oldTangentLambda = vt / 6.0
    ti_m = max(-maxFriction, min(oldTangentLambda, maxFriction))
    dw = ti_m * 2.0 / dc.STONE_RADIUS
            
    # update
    awv1 = np.hypot(v1n + ni_m, v1t + ti_m)
    awth1 = np.arctan2(v1n + ni_m, v1t + ti_m) + th_col
    aww1 = w1 + dw
    awvxy1 = (awv1 * math.sin(awth1), awv1 * math.cos(awth1))
            
    awv0 = np.hypot(v0n - ni_m, v0t - ti_m)
    awth0 = np.arctan2(v0n - ni_m, v0t - ti_m) + th_col
    aww0 = w0 + dw
    awvxy0 = (awv0 * math.sin(awth0), awv0 * math.cos(awth0))

    return awvxy0, aww0, awvxy1, aww1

# whole simulation

# asleep stone
# 0 : index
# 1 : (x, y)

# awake stone
# 0 : index
# 1 : (x, y) current
# 2 : (vx, vy)
# 3 : w
# 4 : (gx, gy) if stop

def deliver_by_xy_vxyw(stones, index, oxy, vxy, w):
    gxy = calc_xy_by_xy_vxyw(oxy, vxy, w)
    if not len(stones):
        # no asleep stones
        return [(index, gxy)]
    asleep = copy.deepcopy(stones)
    awake = [(index, oxy, vxy, w, gxy)]
    t = 0
    while len(awake):
        print(t, awake, asleep)
        # awake itself
        dt_self = []
        for aw in awake:
            st = calc_t_by_v(dc.calc_v(aw[2]))
            dt_self.append(st)
        # awake - awake
        dt_awake_min = float('inf')
        for i, aw0 in enumerate(awake):
            for j, aw1 in enumerate(awake[:i]):
                if is_collisious_aw_aw(aw0[1], aw0[2], aw1[1], aw1[2]):
                    dt = calc_dt_by_aw_aw(aw0[1], aw0[2], aw0[3], aw0[4],
                                          aw1[1], aw1[2], aw1[3], aw1[4])
                    if dt < dt_awake_min:
                        dt_awake_min = dt
                        pair_awake = (i, j)
        # awake - asleep
        dt_asleep_min = float('inf')
        for i, aw in enumerate(awake):
            for j, asl in enumerate(asleep):
                if is_collisious_aw_as(aw[1], aw[1], asl[1]):
                    dt = calc_dt_by_aw_asl(aw[1], aw[2], aw[3], aw[4], asl[1])
                    if dt < dt_asleep_min:
                        dt_asleep_min= dt
                        pair_asleep = (i, j)
        if dt_awake_min <= dt_asleep_min:
            if dt_awake_min < DT_COLLISION:
                # awake - awake collision
                aw0, aw1 = awake[pair_awake[0]], awake[pair_awake[1]]
                awvxy0, aww0, awvxy1, aww1 = collision_aw_aw(aw0[1], aw0[2], aw0[3],
                                                             aw1[1], aw1[2], aw1[3])
                awake[pair_awake[0]] = (aw0[0], aw0[1], awvxy0, aww0, calc_xy_by_xy_vxyw(aw0[1], awvxy0, aww0))
                awake[pair_awake[1]] = (aw1[0], aw1[1], awvxy1, aww1, calc_xy_by_xy_vxyw(aw1[1], awvxy1, aww1))
                continue
            dt_all = dt_awake_min
        else:
            if dt_asleep_min < DT_COLLISION:
                # awake - asleep collision
                aw, asl = awake[pair_asleep[0]], asleep[pair_asleep[1]]
                awvxy, aww, asvxy, asw = collision_aw_asl(aw[1], aw[2], aw[3], asl[1])
                awake[pair_asleep[0]] = (aw[0], aw[1], awvxy, aww, calc_xy_by_xy_vxyw(aw[1], awvxy, aww))
                awake.append((asl[0], asl[1], asvxy, asw, calc_xy_by_xy_vxyw(asl[1], asvxy, asw)))
                asleep.pop(pair_asleep[1])
                continue
            dt_all = dt_asleep_min
        print("dt_all = %f" % dt_all)
        next_awake = []
        # stop
        for i, st in enumerate(dt_self):
            if st <= dt_all:
                aw = awake[i]
                asleep.append((aw[0], aw[4]))
            else:
                next_awake.append(aw)
        print(next_awake)
        # move
        for i, aw in enumerate(next_awake):
            nxy, nvxy = step_xy_vxy_by_xy_vxywt(aw[1], aw[2], aw[3], dt_all)
            next_awake[i] = (aw[0], nxy, nvxy, aw[3], aw[4])
        awake = next_awake
        t += dt
    return asleep

def deliver_by_vxyw_from_throw(stones, index, vxy, w):
    return deliver_by_xy_vxyw(stones, index, dc.XY_THROW, vxy, w)

if __name__ == '__main__':

    # test
    print("DR_V_R =", DR_V_R)
    
    a = ALPHA
    print("ALPHA =", ALPHA)
    print("B =", math.cos(a) / math.sqrt(1 - (math.cos(a) ** 2)))
    b = B
    print(math.acos(b / math.sqrt(b * b + 1)))

    print(PHI)

    print(A)

    print(calc_r_by_theta(PHI))

    print(calc_theta_by_r(dc.calc_r(dc.XY_TEE, dc.XY_THROW)))

    print(calc_v_by_theta(1.53))

    print(calc_vtheta_by_v(31.0))
    
    print("*** Draw Shot ***")

    draw_vxy = calc_vxy_by_xyw_from_throw(dc.XY_TEE, dc.W_SPIN)
    print("Draw Shot(R) = ", draw_vxy)
    print("Simulation Result of Draw Shot(R) =", calc_xy_by_vxyw_from_throw(draw_vxy, dc.W_SPIN))

    draw_vxy = calc_vxy_by_xyw_from_throw(dc.XY_TEE, -dc.W_SPIN)
    print("Draw Shot(L) = ", draw_vxy)
    print("Simulation Result of Draw Shot(L) =", calc_xy_by_vxyw_from_throw(draw_vxy, -dc.W_SPIN))

    guard = (dc.X_TEE + 1.0, dc.Y_TEE - 3.0)
    guard_vxy = calc_vxy_by_xyw_from_throw(guard, dc.W_SPIN)
    print("right guard position =", guard)
    print("Guard Shot(R) =", guard_vxy)
    print("Simulation Result of Draw Shot(R) =", calc_xy_by_vxyw_from_throw(guard_vxy, dc.W_SPIN))

    guard_vxy = calc_vxy_by_xyw_from_throw(guard, -dc.W_SPIN)
    print("right guard position =", guard)
    print("Guard Shot(L) =", guard_vxy)
    print("Simulation Result of Draw Shot(L) =", calc_xy_by_vxyw_from_throw(guard_vxy, -dc.W_SPIN))

    print("%f < %f < %f" % (calc_r_lb_by_lv(36, 33), calc_r_by_lv(36, 33), calc_r_ub_by_lv(36, 33)))

    print("Hit Shot by velocity")

    target_xy = dc.XY_TEE
    print("target stone position =", target_xy)
    #print(calc_r_by_lv(dc.calc_r(dc.XY_THROW, target_xy), 33.0))
    hit_vxy = calc_vxy_by_xylvw_from_throw(target_xy, 33.0, dc.W_SPIN)
    print("Hit Shot(R) =", hit_vxy)

    print("target stone position =", target_xy)
    hit_vxy = calc_vxy_by_xylvw_from_throw(target_xy, 33.0, -dc.W_SPIN)
    print("Hit Shot(L) =", hit_vxy)


    target_xy = (dc.X_TEE + 1.0, dc.Y_TEE - 3.0)
    print("target stone position =", target_xy)
    #print(calc_r_by_lv(dc.calc_r(dc.XY_THROW, target_xy), 33.0))
    hit_vxy = calc_vxy_by_xylvw_from_throw(target_xy, 33.0, dc.W_SPIN)
    print("Hit Shot(R) =", hit_vxy)
    
    print("target stone position =", target_xy)
    hit_vxy = calc_vxy_by_xylvw_from_throw(target_xy, 33.0, -dc.W_SPIN)
    print("Hit Shot(L) =", hit_vxy)

    print("Hit Shot by distance to stop")

    target_xy = dc.XY_TEE
    print("target stone position =", target_xy)
    #print(calc_r_by_lv(dc.calc_r(dc.XY_THROW, target_xy), 33.0))
    print("Hit Shot(R) =", calc_vxy_by_xylrw_from_throw(target_xy, 10.0, dc.W_SPIN))

    print("target stone position =", target_xy)
    print("Hit Shot(L) =", calc_vxy_by_xylrw_from_throw(target_xy, 10.0, -dc.W_SPIN))
    
    
    target_xy = (dc.X_TEE + 1.0, dc.Y_TEE - 3.0)
    print("target stone position =", target_xy)
    #print(calc_r_by_lv(dc.calc_r(dc.XY_THROW, target_xy), 33.0))
    print("Hit Shot(R) =", calc_vxy_by_xylrw_from_throw(target_xy, 10.0, dc.W_SPIN))
    
    print("target stone position =", target_xy)
    print("Hit Shot(L) =", calc_vxy_by_xylrw_from_throw(target_xy, 10.0, -dc.W_SPIN))

    print(np.array([[((calc_r_by_lv(l, v) - calc_r_lb_by_lv(l, v)) / (calc_r_ub_by_lv(l, v) - calc_r_lb_by_lv(l, v)))
                     for l in range(28, 38, 2)] for v in range(30, 33)]))

    print(np.array([[1 - (calc_r_lb_by_lv(l, v) / calc_r_by_lv(l, v))
                     for l in range(28, 38, 2)] for v in range(30, 33)]))

    print(np.array([[(calc_r_ub_by_lv(l, v) / calc_r_by_lv(l, v)) - 1
                     for l in range(28, 38, 2)] for v in range(30, 33)]))


    print("step function")
    gxy = dc.XY_TEE
    w = dc.W_SPIN
    xy = dc.XY_THROW
    vxy = calc_vxy_by_xyw(xy, gxy, w)
    t = calc_t_by_v(dc.calc_v(vxy))
    
    x = [float(i) for i in range(2, 100)]
    print(vxy)
    steps = 0
    while t > 0.0001:
        print(t, xy, vxy)
        dt = t * (1 - 1 / x[steps])
        xy, vxy = step_xy_vxy_by_xy_vxywt(xy, vxy, w, dt)
        t -= dt
        steps += 1
    print(xy, vxy)


    w = -dc.W_SPIN
    xy = dc.XY_THROW
    vxy = calc_vxy_by_xyw(xy, gxy, w)
    t = calc_t_by_v(dc.calc_v(vxy))

    print(vxy)
    steps = 0
    while t > 0.0001:
        print(t, xy, vxy)
        dt = t * (1 - 1 / x[steps])
        xy, vxy = step_xy_vxy_by_xy_vxywt(xy, vxy, w, dt)
        t -= dt
        steps += 1
    print(xy, vxy)

    print("*** whole simulation ***")
    # null board
    asleep = []
    draw_vxy = calc_vxy_by_xyw_from_throw(dc.XY_TEE, dc.W_SPIN)
    print(deliver_by_vxyw_from_throw(asleep, 0, draw_vxy, dc.W_SPIN))

    # 1 stone
    target_xy = dc.XY_TEE
    asleep = [(1, target_xy)]
    hit_vxy = calc_vxy_by_xylvw_from_throw(target_xy, 33.0, dc.W_SPIN)
    print(deliver_by_vxyw_from_throw(asleep, 0, hit_vxy, dc.W_SPIN))
