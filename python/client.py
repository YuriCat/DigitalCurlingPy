# -*- coding: utf-8 -*-
# client.py
# Katsuki Ohto

import sys
import socket
from contextlib import closing

import dc
#import search

DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 9876

def send_msg(sock, msg):
    buf = msg.encode('utf-8')
    print(buf)
    sock.send(buf)

def recv_msg(sock, buf_size):
    msg = sock.recv(buf_size)
    print(msg)
    return msg.decode('utf-8')

class Client:
    def __init__(self, sock):
        self.sock = sock
        self.board = dc.Board()
        self.my_first_color = -1
        self.my_color = -1
        self.ends = -1
        self.score = [0 for _ in range(dc.N_ENDS)]
        self.time = [0, 0]
        self.action_dct = {
            'CONNECTED': [(self.nothing, [])],
            'LOGIN' : [(self.nothing, [])],
            'ISREADY' : [(self.init_game, []),
                         (self.init_end, []),
                         (self.send_ready, [])],
            'SETSTATE' : [(self.set_state, [i for i in range(4)])],
            'POSITION' : [(self.set_stone, [i for i in range(2 * dc.N_STONES)])],
            'GO': [(self.play, [i for i in range(2)])],
            'GAMEOVER' : [(self.close_game, [])],
            'DISCONNECT' : [(self.close_all, [])]}

    def nothing(self, arg_list):
        return 0

    def init_all(self, arg_list):
        print('Client.init_all()')
        return 0
    
    def init_game(self, arg_list):
        print('Client.init_game()')
        return 0
    
    def init_end(self, arg_list):
        print('Client.init_end()')
        self.board.init()
        return 0
    
    def close_end(self, arg_list):
        print('Client.close_end()')
        return 0
    
    def close_game(self, arg_list):
        print('Client.close_game()')
        return 0
    
    def close_all(self, arg_list):
        print('Client.close_all()')
        return -1
    
    def send_ready(self, arg_list):
        print('Client.send_ready()')
        send_msg(self.sock, 'READYOK')
        return 0
    
    def set_state(self, arg_list):
        print('Client.set_state()')
        
        if self.ends == -1:
            self.ends = int(arg_list[2])
            self.score = [0 for _ in range(self.ends)]
        
        if self.my_first_color == -1:
            if arg_list[3] == 0:
                self.my_first_color = self.my_color = dc.BLACK
            else:
                self.my_first_color = self.my_color = dc.WHITE
    
        self.board.end = int(arg_list[1])
        self.board.turn = int(arg_list[0])
        return 0

    def set_stone(self, arg_list):
        print('Client.set_stone()')
        print(arg_list)
        for i in range(0, len(arg_list), 2):
            p = (float(arg_list[i]), float(arg_list[i + 1]))
            self.board.stone[i // 2] = p
        return 0
    
    def play(self, arg_list):
        print('Client.play()')
        self.time[dc.BLACK] = int(arg_list[0])
        self.time[dc.WHITE] = int(arg_list[1])
        shot = dc.TEE_SHOT_R
        shot_str = ' '.join(['BESTMOVE', str(float(shot[0])), str(float(shot[1])), str(int(shot[2]))])
        print(shot_str)
        send_msg(self.sock, shot_str)
        return 0
    
    def do_command(self, cmd, arg_list):
        if cmd == "":
            return 0
        ucmd = cmd.upper()
        if not ucmd in self.action_dct:
            print('unknown command')
            return -1
        for act in self.action_dct[ucmd]:
            err = act[0]([arg_list[i] for i in act[1]])
            if err < 0:
                return -2
        return 0

def main():
    
    host = DEFAULT_HOST
    port = DEFAULT_PORT
    buf_size = 1024
    
    id = 'py_sample'
    pwd = 'py_sample_pass'
    name = 'PySample'
    
    client = Client(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
    
    with closing(client.sock):
        client.sock.connect((host, port))
        send_msg(client.sock, ' '.join(['LOGIN', id, pwd, name]))
        while True:
            rmsg_str = recv_msg(client.sock, buf_size)
            rmsg_list = str(rmsg_str).split('\0')
            for rmsg in rmsg_list:
                print(rmsg)
                cmd_list = rmsg.split(' ')
                if len(cmd_list) == 0:
                    break
                cmd = cmd_list[0]
                del cmd_list[0]
                print(cmd)
                err = client.do_command(cmd, cmd_list)
                if err == -2:
                    return
                elif err == -1:
                    i = 1
    return

if __name__ == '__main__':
    main()


