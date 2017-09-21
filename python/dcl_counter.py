# -*- coding: utf-8 -*-
# dcl_counter.py
# Katsuki Ohto

import argparse
import glob
import numpy as np

draw_rate = 0.781

def wl_to_wnwd(r):
    return r[0] + r[1] * draw_rate + r[2] * (1 - draw_rate)

def wl_to_wlwd(r):
    if r[0] + r[1] + r[2] + r[3] <= 0:
        return 0.5
    else:
        return wl_to_wnwd(r) / (r[0] + r[1] + r[2] + r[3])

def analyze_logs(logs, n):
    # result survey
    wl = {}
    first_end_scorer_wl = [[0, 0, 0, 0], [0, 0, 0, 0]]
    scores = {}
    # simulation survey
    errors = [[], []]
    
    log_file_names = glob.glob(logs)
    if n is not None:
        # sort logs by record time
        log_file_names = sorted(log_file_names, key = lambda log : int(log.split('[')[1].split(']')[0]))
        log_file_names = log_file_names[0 : int(n)]
    
    #for log in log_file_names:
        #print(log)
    
    for f in log_file_names:
        # get data
        name = ["", ""]
        chosen = (0, 0)
        run = (0, 0)
        score = [-1, -1]
        flip = 1
        end = 0
        first_end_scorer = -1
        
        for line in open(f, 'r'):
            data = line.split()
            if 'First=' in data[0]:
                name[0] = data[0][6:]
            elif 'Second=' in data[0]:
                name[1] = data[0][7:]
            elif 'BESTSHOT' in data[0]:
                chosen = (float(data[1]), float(data[2]))
            elif 'RUNSHOT' in data[0]:
                run = (float(data[1]), float(data[2]))
                errors[0].append(run[0] - chosen[0])
                errors[1].append(run[1] - chosen[1])
            elif 'TOTALSCORE' in data[0]:
                score[0] = int(data[1])
                score[1] = int(data[2])
            elif 'SCORE' in data[0]:
                if flip * int(data[1]) < 0:
                    flip = -flip
                if end == 0 and np.abs(int(data[1])) == 1:
                    first_end_scorer = 0 if int(data[1]) > 0 else 1
                end += 1
    
        for i in range(2):
            if name[i] not in wl:
                wl[name[i]] = [[0, 0, 0, 0], [0, 0, 0, 0]]
                scores[name[i]] = [[0, 0], [0, 0]]

        for i in range(2):
            for c in range(2):
                scores[name[i]][i][i ^ c] += score[c]

        if score[0] > score[1]:
            wl[name[0]][0][0] += 1
            wl[name[1]][1][3] += 1
        elif score[0] < score[1]:
            wl[name[0]][0][3] += 1
            wl[name[1]][1][0] += 1
        else:
            if flip == -1:
                wl[name[0]][0][1] += 1
                wl[name[1]][1][2] += 1
            else:
                wl[name[0]][0][2] += 1
                wl[name[1]][1][1] += 1

        if first_end_scorer != -1:
            if score[first_end_scorer] > score[1 - first_end_scorer]:
                first_end_scorer_wl[first_end_scorer][0] += 1
            elif score[first_end_scorer] < score[1 - first_end_scorer]:
                first_end_scorer_wl[first_end_scorer][3] += 1
            else:
                if flip == 1 - first_end_scorer * 2:
                    first_end_scorer_wl[first_end_scorer][2] += 1
                else:
                    first_end_scorer_wl[first_end_scorer][1] += 1

    print(wl)
    print(scores)
    
    print(first_end_scorer_wl)

    wn_with_draw = {}
    
    for nm, r in wl.items():
        wn_with_draw[nm] = [wl_to_wnwd(r[0]), wl_to_wnwd(r[1])]
    
    print(wn_with_draw)

    wl_with_draw = {}

    for nm, r in wl.items():
        wl_with_draw[nm] = [wl_to_wlwd(r[0]), wl_to_wlwd(r[1])]
    
    print(wl_with_draw)

    wr_with_draw = {}
    
    for nm, r in wl_with_draw.items():
        wr_with_draw[nm] = (r[0] + r[1]) / 2

    print(wr_with_draw)

    print("error in Vx : mean = %f stddev = %f" % (np.mean(errors[0]), np.std(errors[0])))
    print("error in Vy : mean = %f stddev = %f" % (np.mean(errors[1]), np.std(errors[1])))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--logs', required=True)
    parser.add_argument('--n', required=False)
    args = parser.parse_args()
    
    analyze_logs(args.logs, args.n)


