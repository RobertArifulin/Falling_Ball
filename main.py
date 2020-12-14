import pygame as pg
import random
import types as tp
import Button as bt

pg.init()
pg.font.init()
f1 = pg.font.Font(None, 24)
text_m = f1.render('m =', False, (0, 0, 0))
text_V = f1.render('V =', False, (0, 0, 0))
text_h = f1.render('h =', False, (0, 0, 0))

screen_w = pg.display.Info().current_w
screen_h = pg.display.Info().current_h

screen = pg.display.set_mode((tp.WIDTH, tp.HEIGHT))
pg.display.set_caption("Falling Ball")
clock = pg.time.Clock()

size = random.randint(1, 50)
screen.fill((255, 255, 255))

run = True
while run:
    pg.time.delay(200)
    size = random.randint(1, 50)
    bt.draw_text_input(screen)
    screen.blit(text_m, (tp.Button_x, tp.Button_y * 5 - 15))
    screen.blit(text_V, (tp.Button_x, tp.Button_y * 10 - 15))
    screen.blit(text_h, (tp.Button_x, tp.Button_y * 15 - 15))
    for event in pg.event.get():
        if event.type == pg.QUIT:
            run = False

    pg.display.flip()
    clock.tick(tp.FPS)
    pg.display.update()
