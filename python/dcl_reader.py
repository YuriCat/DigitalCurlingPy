# -*- coding: utf-8 -*-
# dcl_reader.py
# Katsuki Ohto

import copy
import glob
import argparse
import numpy as np

import dc

def read_dcl(dcl_file):
    
    # get data
    game_record = {}
    game_record['name'] = ["", ""]
    game_record['end'] = []
    game_record['score'] = 0
    game_record['path'] = dcl_file
        
    end_record = {'turn':[]}
    
    turn_record = {}

    chosen = (0, 0)
    run = (0, 0)
    score = 0
    turn = 0
    end = 0
    first_end_scorer = -1
    stone = np.zeros(dc.N_STONES, dtype = tuple)
        
    for line in open(dcl_file, 'r'):
        data = line.split()
        if 'First=' in data[0]:
            game_record['name'][0] = data[0][6:]
        elif 'Second=' in data[0]:
            game_record['name'][1] = data[0][7:]
        elif 'POSITION' in data[0]:
            for i in range(dc.N_STONES):
                stone[dc.N_STONES - 1 - i] = dc.official_to_ayumu_position((float(data[1 + i * 2]),
                                                          float(data[1 + i * 2 + 1])))
        elif 'BESTSHOT' in data[0]:
            chosen = (float(data[1]), float(data[2]))
        elif 'RUNSHOT' in data[0]:
            run = (float(data[1]), float(data[2]))
            
            # proceed turn
            turn_record['stone'] = copy.deepcopy(stone)
            turn_record['chosen'] = copy.deepcopy(chosen)
            turn_record['run'] = copy.deepcopy(run)
            
            end_record['turn'].append(copy.deepcopy(turn_record))
            turn_record = {}
            
            turn += 1

        elif 'TOTALSCORE' in data[0]:
            game_score = (int(data[1]), int(data[2]))
            game_record['score'] = game_score[0] + game_score[1]
        elif 'SCORE' in data[0]:
            score = int(data[1])
            
            # proceed end
            end_record['score'] = score
            game_record['end'].append(end_record)

            end_record = {'turn':[]}

    return game_record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--logs', required=True)
    parser.add_argument('--n', required=False)
    args = parser.parse_args()
    
    analyze_logs(args.logs, args.n)


