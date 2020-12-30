import string
WIDTH = 1600  # ширина игрового окна
HEIGHT = 900  # высота игрового окна
FPS = 60  # частота кадров в секунду
Button_x = 15  # х коодината
Button_y = 15  # у коодината
Button_size = (200, 50)  # размер
Input_size = (100, 50)

white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)


title_m = 'm(кг)'
title_V = 'V(м^3)'
title_h = 'h(м)'

limit = 10
white_list = []
for i in string.digits:
    white_list.append(i)
white_list.append('.')
white_list.append(',')

