# -*- coding: utf-8 -*-
# play.py
# Katsuki Ohto

import numpy as np

import dc
import ayumu_dc as adc

import simulator

def evaluate(board):
    # convert board info to image
    img = np.array(28 * 28, )

def search(board):
    e = board.end
    t = board.turn
    c = dc.to_turn_color(t)
    bestEval = -9999
    bestmove = dc.TEE_SHOT_R
    
    mesh = np.empty((adc.GRID_WIDTH + 1, adc.GRID_LENGTH + 1), dtype = np.float32)
    
    for s in range(2):
        for w in range(adc.GRID_WIDTH + 1):
            for l in range(adc.GRID_LENGTH + 1):
                mv = (adc.WtoVX(w), adc.LtoVY(l), s)
                bd = board
                after_bd = simulator.makeMoveNoRand_PA(bd, t, mv)
                sc = dc.count_score(after_bd)
                wp = adc.wp_table[e][dc.StoIDX(sc)]
                wp = -wp if c == WHITE
                mesh[w][l] = wp
    
        # integrate
        for w in range(ROOT_LAST_GRID_ZW, ROOT_LAST_GRID_W - ROOT_LAST_GRID_ZW + 1, 1):
            for l in range(ROOT_LAST_GRID_ZL, ROOT_LAST_GRID_L - ROOT_LAST_GRID_ZL + 1, 1):
                iev = 0;
                for dw in range(-ROOT_LAST_GRID_ZW, ROOT_LAST_GRID_ZW + 1, 1):
                    for dl in range(-ROOT_LAST_GRID_ZL, ROOT_LAST_GRID_ZL + 1, 1):
                        pdf = adc.grid_pdf_table[abs(dw)][abs(dl)];
                        iev += pdf * mesh[w + dw][l + dl];
                        if iev > bestEval:
                            bestEval = iev
                            bestMove.set(lastWtoVX(w), lastLtoVY(l), s);
                                                                                                }
                                                                                                }
                                                                                                }
        }

def play(board):
    mv = search(board)
    return mv
