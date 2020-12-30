import pygame as pg
import Fall_Types as tp
# import matplotlib.pyplot as plt
import string
import pygame_gui as pgui


def get_p(ph, pv, pm):
    ch = ''
    cv = ''
    cm = ''
    for i in ph:
        if i == ',':
            ch += '.'
        else:
            ch += i
    for i in pv:
        if i == ',':
            cv += '.'
        else:
            cv += i
    for i in pm:
        if i == ',':
            cm += '.'
        else:
            cm += i
    while ch.count('.') > 1:
        ch.replace('.', '', 1)
    while cv.count('.') > 1:
        cv.replace('.', '', 1)
    while cm.count('.') > 1:
        cm.replace('.', '', 1)
    return ch, cv, cm


'''Создаем окно, настраиваем шрифт и часы'''
pg.init()
pg.font.init()
f1 = pg.font.Font(None, 30)
manager = pgui.UIManager((tp.WIDTH, tp.HEIGHT))
screen = pg.display.set_mode((tp.WIDTH, tp.HEIGHT))
pg.display.set_caption("Falling Ball")
clock = pg.time.Clock()
screen.fill(tp.white)

'''Предварительно задаем переменные'''
h = 200000
V = 10
m = 10
text_m = str(m)
text_V = str(V)
text_h = str(h)
running_time = False

'''Создаем начальные элементы: кнопку и текстовое поле'''
Start_b = pgui.elements.UIButton(relative_rect=pg.Rect((50, 10), tp.Button_size),
                                 text='Start',
                                 manager=manager)
m_entry = pgui.elements.UITextEntryLine(relative_rect=pg.Rect((120, 100), tp.Input_size),
                                        manager=manager)
V_entry = pgui.elements.UITextEntryLine(relative_rect=pg.Rect((120, 170), tp.Input_size),
                                        manager=manager)
h_entry = pgui.elements.UITextEntryLine(relative_rect=pg.Rect((120, 240), tp.Input_size),
                                        manager=manager)

'''Задаем начальный текст'''
m_entry.set_text(text_m)
V_entry.set_text(text_V)
h_entry.set_text(text_h)

'''Максимальная длина строки - 10'''
m_entry.set_text_length_limit(tp.limit)
V_entry.set_text_length_limit(tp.limit)
h_entry.set_text_length_limit(tp.limit)

'''Разрешается вписывать только цифры и . , '''
m_entry.set_allowed_characters(tp.white_list)
V_entry.set_allowed_characters(tp.white_list)
h_entry.set_allowed_characters(tp.white_list)

'''Пишем поясняющие заголовки'''
screen.blit(f1.render(tp.title_m, False, tp.black), (50, 105))
screen.blit(f1.render(tp.title_V, False, tp.black), (50, 175))
screen.blit(f1.render(tp.title_h, False, tp.black), (50, 245))

run = True
while run:
    h = m_entry.get_text()
    V = m_entry.get_text()
    m = m_entry.get_text()
    time_delta = clock.tick(60) / 1000.0
    for event in pg.event.get():
        if event.type == pg.QUIT:
            run = False

        if event.type == pg.USEREVENT:
            if event.user_type == pgui.UI_BUTTON_PRESSED:
                if event.ui_element == Start_b:
                    if Start_b.text == 'Start':
                        Start_b.set_text('Stop')
                        running_time = True
                    else:
                        Start_b.set_text('Start')
                        running_time = False

        manager.process_events(event)

    # if not m_entry.is_focused:
    #     a = get_p(h_entry.text, V_entry.text, m_entry.text)
    #     m_entry.set_text(a[2])
    # if not V_entry.is_focused:
    #     a = get_p(h_entry.text, V_entry.text, m_entry.text)
    #     V_entry.set_text(a[1])
    # if not h_entry.is_focused:
    #     a = get_p(h_entry.text, V_entry.text, m_entry.text)
    #     h_entry.set_text(a[0])
    manager.update(time_delta)
    manager.draw_ui(screen)

    clock.tick(tp.FPS)
    pg.display.update()
