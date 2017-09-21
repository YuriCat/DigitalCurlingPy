# -*- coding: utf-8 -*-
# dcl_statistics.py
# Katsuki Ohto

# analyze statistics on game (.dcl)

import copy
import glob
import argparse
import numpy as np

import dc
import dcl_reader

debug_flag = False

def debug_print(a):
    if debug_flag:
        print(a)

def debug_stop():
    if debug_flag:
        input()

def analyze(logs):

    first_turn_pass_count = 0
    black_ops_stone_takeout_count = 0
    black_my_stone_add_count = 0
    white_ops_stone_takeout_count = 0
    white_my_stone_add_count = 0
    
    for log in logs:
        debug_print(log)
        #input()
        
        game_first_turn_pass_count = 0
        game_black_ops_stone_takeout_count = 0
        game_black_my_stone_add_count = 0
        game_white_ops_stone_takeout_count = 0
        game_white_my_stone_add_count = 0
        
        if len(log['end']) > 0:
            if dc.count_in_play_area_a(log['end'][0]['turn'][1]['stone']) == 0:
                game_first_turn_pass_count += 1
            
            for t in range(0, dc.N_TURNS, 2):
                org_cnt = dc.count_in_house_a(log['end'][0]['turn'][t]['stone'], dc.WHITE)
                next_cnt = dc.count_in_house_a(log['end'][0]['turn'][t + 1]['stone'], dc.WHITE)
                if next_cnt < org_cnt:
                    game_black_ops_stone_takeout_count += 1
        
            for t in range(0, dc.N_TURNS, 2):
                org_cnt = dc.count_in_house_a(log['end'][0]['turn'][t]['stone'], dc.BLACK)
                next_cnt = dc.count_in_house_a(log['end'][0]['turn'][t + 1]['stone'], dc.BLACK)
                if next_cnt > org_cnt:
                    game_black_my_stone_add_count += 1
            
            for t in range(1, dc.N_TURNS - 1, 2):
                org_cnt = dc.count_in_house_a(log['end'][0]['turn'][t]['stone'], dc.BLACK)
                next_cnt = dc.count_in_house_a(log['end'][0]['turn'][t + 1]['stone'], dc.BLACK)
                if next_cnt < org_cnt:
                    game_white_ops_stone_takeout_count += 1
            
            for t in range(1, dc.N_TURNS - 1, 2):
                org_cnt = dc.count_in_house_a(log['end'][0]['turn'][t]['stone'], dc.WHITE)
                next_cnt = dc.count_in_house_a(log['end'][0]['turn'][t + 1]['stone'], dc.WHITE)
                if next_cnt > org_cnt:
                    game_white_my_stone_add_count += 1
    
        debug_print(game_first_turn_pass_count)
        debug_print(game_black_ops_stone_takeout_count)
        debug_print(game_black_my_stone_add_count)
        debug_print(game_white_ops_stone_takeout_count)
        debug_print(game_white_my_stone_add_count)
        debug_stop()

        first_turn_pass_count += game_first_turn_pass_count
        black_ops_stone_takeout_count += game_black_ops_stone_takeout_count
        black_my_stone_add_count += game_black_my_stone_add_count
        white_ops_stone_takeout_count += game_white_ops_stone_takeout_count
        white_my_stone_add_count += game_white_my_stone_add_count

    print("first_turn_pass_count = ", first_turn_pass_count)
    print("black_ops_stone_takeout_count = ", black_ops_stone_takeout_count)
    print("black_my_stone_add_count = ", black_my_stone_add_count)
    print("white_ops_stone_takeout_count = ", white_ops_stone_takeout_count)
    print("white_my_stone_add_count = ", white_my_stone_add_count)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--logs', required=True)
    parser.add_argument('--n', required=False)
    args = parser.parse_args()
    
    logs = []
    for log_file in glob.glob(args.logs):
        print(log_file)
        logs.append(dcl_reader.read_dcl(log_file))

    analyze(logs)


