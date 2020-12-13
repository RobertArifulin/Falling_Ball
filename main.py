import pygame as pg
WIDTH = 360  # ширина игрового окна
HEIGHT = 480 # высота игрового окна
FPS = 30 # частота кадров в секунду

pg.init()
pg.font.init()

screen = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("Falling Ball")
clock = pg.time.Clock()

print('hello')

run = True
while run:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            run = False
    pg.time.delay(50)
    screen.fill((0, 0, 0))
    pg.display.flip()
