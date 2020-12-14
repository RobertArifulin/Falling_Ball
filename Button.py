import types as tp
import pygame as pg


def draw_text_input(win):
    for i in range(1, 4):
        pg.draw.rect(win, (100, 100, 100), (tp.Button_x, tp.Button_y * 5 * i, tp.Button_size * 4, tp.Button_size))  # рисуем поле для ввода параметров
        pg.draw.rect(win, (200, 200, 200), (tp.Button_x + 2, tp.Button_y * 5 * i + 2, tp.Button_size * 4 - 4, tp.Button_size - 4))


def mouse_pos(Event):
    if Event.type == pg.MOUSEBUTTONDOWN and Event.button == 1:
        x = Event.pos[0]
        y = Event.pos[1]

    if tp.Button_x < x < tp.Button_x + tp.Button_size * 4 and tp.Button_y * 5 < y < tp.Button_y + tp.Button_size:
        tp.mouse_pos = '1_text'
    if tp.Button_x < x < tp.Button_x + tp.Button_size * 4 and tp.Button_y * 10 < y < tp.Button_y * 10 + tp.Button_size:
        tp.mouse_pos = '2_text'
    if tp.Button_x < x < tp.Button_x + tp.Button_size * 4 and tp.Button_y * 15 < y < tp.Button_y * 15 + tp.Button_size:
        tp.mouse_pos = '3_text'
