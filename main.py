import pygame as pg
import Fall_Types as tp
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import math
import pygame_gui as pgui
from matplotlib.animation import FuncAnimation

# суммарное ускорение от g и сопр. среды
def a_F(new_g, new_Fr):
    global m
    new_a = (new_g * m - new_Fr) / m
    return round(new_a, 4)


# скорость от начальной v и ускорения
def v_S(new_a, new_v0, new_S):
    global m
    # if new_S * 2 * new_a + new_v0 ** 2 < 0:
    #     new_a = 0
    #     new_v0 = 0
    new_v = math.sqrt(new_S * 2 * new_a + new_v0 ** 2)
    return round(new_v, 4)


def S_t(new_v0, new_t, new_a):
    new_S = new_v0 * new_t + new_a * (new_t ** 2) / 2
    return round(new_S, 6)


# сила тяжести
def g_force(new_m, new_g):
    new_F = new_m * new_g
    return round(new_F, 4)


# находит g по высоте
def g_h(new_h):
    new_g = tp.G * (tp.M / (tp.R + new_h) ** 2)
    return round(new_g, 4)


# находит плотность воздуха по высоте
def p_h(new_h):
    i = len(tp.ar_h) - 1
    while tp.ar_h[i] >= new_h:
        i -= 1
    new_p = tp.ar_p[i] - (new_h - tp.ar_h[i]) / (tp.ar_h[i + 1] - tp.ar_h[i]) * (tp.ar_p[i] - tp.ar_p[i + 1])
    return round(new_p, 6)


# давление под водой
def P_h(new_h):
    new_P = tp.water_p * tp.sea_g * new_h
    return round(new_P, 4)


# сила сопротивления среды
def resist_force(new_v, new_p, new_S, new_Cf):
    new_Fr = new_Cf * new_p * (new_v ** 2) * new_S / 2
    return round(new_Fr, 6)


# к сжатия воздуха от глубины
def k_h(new_h):
    new_P = P_h(new_h)
    if new_P <= tp.ar_P[0]:
        return tp.ar_k[0]
    if new_P >= tp.ar_P[-1]:
        return tp.ar_k[-1]
    i = len(tp.ar_k) - 1
    while tp.ar_P[i] > new_P:
        i -= 1
    if new_P <= (tp.ar_P[i] + tp.ar_P[i + 1]) / 2:
        return tp.ar_k[i]
    else:
        return tp.ar_k[i + 1]


# сжатие шара на глубине
def comress_h(new_h, old_V):
    new_k = k_h(new_h)
    new_P = P_h(new_h) / 101325
    new_V = old_V/(new_P * new_k)
    return round(new_V, 7)


# поиск объема сегмента шара
def Vseg_h(new_h, new_r):
    new_Vseg = math.pi * (new_h ** 2) * (3 * new_r - new_h) / 3
    return new_Vseg


# сила Архимеда
def Arh_force(new_V):
    new_Farh = tp.water_p * tp.sea_g * new_V
    return new_Farh


# проверка нажатия
def focus_check():
    global check, m_entry, V_entry, h_entry, h_delta_entry
    if m_entry.is_focused:
        check[0] = 1
    if V_entry.is_focused:
        check[1] = 1
    if h_entry.is_focused:
        check[2] = 1
    if h_delta_entry.is_focused:
        check[3] = 1


# внесение изменений в параметры после исправлений
def changes():
    global check, m_entry, V_entry, h_entry, h, m, V, h_delta_entry
    if not m_entry.is_focused and check[0] == 1:
        m_entry.set_text(limit_check(m_entry.get_text(), '1', '1', '1')[0])
        m = float(m_entry.get_text())
        check[0] = 0
    if not V_entry.is_focused and check[1] == 1:
        V_entry.set_text(limit_check('1', V_entry.get_text(), '1', '1')[1])
        V = float(V_entry.get_text())
        check[1] = 0
    if not h_entry.is_focused and check[2] == 1:
        h_entry.set_text(limit_check('1', '1', h_entry.get_text(), '1')[2])
        h = float(h_entry.get_text())
        check[2] = 0
    if not h_delta_entry.is_focused and check[3] == 1:
        h_delta_entry.set_text(limit_check('1', '1', '1', h_delta_entry.get_text())[3])
        tp.delta_h = float(h_delta_entry.get_text())
        check[3] = 0


# очистка текста от лишних запятых/точек, замена , -> .
def clean(new_text):
    ctext = ''
    flag = False

    for i in range(0, 10):
        if new_text.find(str(i)) != -1:
            flag = True

    if not flag:
        new_text = '0.00001'

    for i in new_text:
        if i == ',':
            ctext += '.'
        else:
            ctext += i

    new_text = ctext.strip('.').lstrip('0')
    ctext = new_text.replace('.', '', (new_text.count('.') - 1))

    return ctext


# вкл/выкл текстовые поля
def enable(yes):
    global m_entry, V_entry, h_entry
    if yes:
        m_entry.enable()
        V_entry.enable()
        h_entry.enable()
        h_delta_entry.enable()
    if not yes:
        m_entry.disable()
        V_entry.disable()
        h_entry.disable()
        h_delta_entry.disable()


# отрисовка текста
def draw_text(window):
    global f1
    screen.blit(f1.render('Ускорение    ' + str(a), False, tp.black), (50, tp.text_y))
    screen.blit(f1.render('Вес  ' + str(weight), False, tp.black), (50, tp.text_y + 70))
    screen.blit(f1.render('Скорось  ' + str(v1), False, tp.black), (50, tp.text_y + 70 * 2))
    screen.blit(f1.render('Высота   ' + str(h), False, tp.black), (50, tp.text_y + 70 * 3))

    window.blit(f1.render('м/с2', False, tp.black), (tp.text_x, tp.text_y))
    window.blit(f1.render('Н', False, tp.black), (tp.text_x, tp.text_y + 70))
    window.blit(f1.render('м/с', False, tp.black), (tp.text_x, tp.text_y + 70 * 2))
    window.blit(f1.render('м', False, tp.black), (tp.text_x, tp.text_y + 70 * 3))


# проверка границ параметров
def limit_check(pm, pv, ph, pdelta):
    pm = float(clean(pm))
    pv = float(clean(pv))
    ph = float(clean(ph))
    pdelta = float(clean(pdelta))

    if pm < tp.min_m:
        pm = tp.min_m
    if pm > tp.max_m:
        pm = tp.max_m

    if pv < tp.min_v:
        pv = tp.min_v
    if pv > tp.max_v:
        pv = tp.max_v

    if ph < tp.min_h:
        ph = tp.min_h
    if ph > tp.max_h:
        ph = tp.max_h

    if pdelta < tp.min_delta:
        pdelta = tp.min_delta
    if pdelta > tp.max_delta:
        pdelta = tp.max_delta

    return str(pm), str(pv), str(ph), str(pdelta)


def arr_append(new_th, new_weight, new_a, new_delta_h):
    global Fly_arr, a0_arr, hit_arr, part_h_arr, h_entry
    if round(new_a, 1) == 0 or len(a0_arr) > 0:
        a0_arr.append(new_weight)
        part_h_arr[1].append(new_th + new_delta_h)
    else:
        Fly_arr.append(new_weight)
        part_h_arr[0].append(new_th + new_delta_h)


def Draw_graph():
    fig = plt.figure()  # настраиваем размер, чтобы не коверкать картинку
    fig = plt.axis([0, max(all_h_arr)*1.1, min(all_weight_arr)*1.1, tp.max_m*20])
    plt.grid(True)
    plt.plot(part_h_arr[0], Fly_arr, color='b')
    if len(part_h_arr[1]) > 0:
        plt.plot([part_h_arr[0][-1], part_h_arr[1][0]], [Fly_arr[-1], a0_arr[0]], color='g')
        plt.plot(part_h_arr[1], a0_arr, color='g')
        if len(part_h_arr[2]) > 0:
            plt.plot([part_h_arr[1][-1], part_h_arr[2][0]], [a0_arr[-1], hit_arr[0]], color='r')
            print('here 1')
    if len(part_h_arr[1]) == 0 and len(part_h_arr[2]) > 0:
            plt.plot([part_h_arr[0][-1], part_h_arr[2][0]], [Fly_arr[-1], hit_arr[0]], color='r')
            print('here 2')
    if len(part_h_arr[2]) > 0:
        plt.plot(part_h_arr[2], hit_arr, color='r')
        if len(part_h_arr[3]) > 0:
            plt.plot([part_h_arr[2][-1], part_h_arr[3][0]], [hit_arr[-1], part_water_ar[0]], color='r')
    if len(part_h_arr[3]) > 0:
        plt.plot(part_h_arr[3], part_water_ar, color='c')
    if len(part_h_arr[4]) > 0 and len(part_h_arr[3]) > 0:
        plt.plot([part_h_arr[3][-1], part_h_arr[4][0]], [part_water_ar[-1], water_ar[0]], color='m')
        plt.plot(part_h_arr[4], water_ar, color='m')
    plt.xlabel("Путь")  # подпишем оси
    plt.ylabel("Вес")
    plt.savefig('Граф')
    plt.show()


def Draw_animation():
    fig, ax = plt.subplots()
    ax = plt.axis([0, max(all_h_arr) * 1.1, 0, tp.max_m * 12])
    plt.xlabel('Путь')
    plt.ylabel('Вес')
    red_dot, = plt.plot([0], [0], 'ro')
    print(len(all_weight_arr), len(all_h_arr))

    def animate(i):
        red_dot.set_data(all_h_arr[i], all_weight_arr[i])
        line = plt.plot(all_h_arr[:i], all_weight_arr[:i], 'b')
        return red_dot,

    ani = FuncAnimation(fig, animate, interval=10, repeat=True, frames=len(all_weight_arr))
    ani.save('try.gif', writer="pillow")
    # plt.close(ax)
    plt.close(fig)


def Reset_Variable():
    global weight, time, p, a, th, g, v0, v1, Vseg, Farh, Fly_arr, a0_arr, hit_arr, h_arr, all_weight_arr, part_h_arr, on_hit, part_water_ar, water_ar, P, R, all_h_arr
    weight = 0  # вес
    time = 0  # время
    p = 0  # плотность воздуха
    a = 0  # ускорение суммарное
    th = 0  # сколько пролетел (м)
    g = 0  # g
    v0 = 0  # начальная скорость
    v1 = 0  # конечная скорость
    Vseg = 0  # объем сегмента шара
    Farh = 0  # сила Архимеда
    R = round(((V * 3) / (math.pi * 4)) ** (1 / 3), 3)
    P = 101325  # давление
    Fly_arr = [0]  # список веса для графика
    a0_arr = []
    hit_arr = []
    h_arr = []  # список высот для графика
    all_weight_arr = []
    part_water_ar = []
    water_ar = []
    part_h_arr = [[0], [], [], [], []]
    all_weight_arr = [0]
    all_h_arr = [0]  # список высот для графика
    on_hit = True


def Restart_bf():
    global running_time, m, V, h, R
    Start_b.set_text('Start')
    running_time = False
    Reset_Variable()
    m = float(m_entry.get_text())
    V = float(V_entry.get_text())
    h = float(h_entry.get_text())
    R = round(((V * 3) / (math.pi * 4)) ** (1 / 3), 3)
    tp.delta_h = float(h_delta_entry.get_text())
    pg.draw.rect(screen, tp.white, (50, tp.text_y, 350, 230))
    draw_text(screen)


def Draw_graph_bf():
    global running_time, m, V, h, R
    running_time = False
    Start_b.disable()
    Restart_b.disable()
    m_entry.disable()
    V_entry.disable()
    h_entry.disable()
    h_delta_entry.disable()

    Start_b.set_text('Start')
    Draw_graph()
    Draw_animation()

    Start_b.enable()
    Restart_b.enable()
    m_entry.enable()
    V_entry.enable()
    h_entry.enable()
    h_delta_entry.enable()


def weight_delay_control(weight_ar):
    if len(weight_ar) > 1:
        if abs(weight_ar[-1] - weight_ar[-2]) >= 10:
            tp.max_delta -= 0.5



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
m = 10
V = 10
h = 200000
R = ((V * 3) / (math.pi * 4)) ** (1 / 3)
r = R
S = math.pi * r ** 2

text = '''
Высота 
Скорость
Ускорение
'''

'''Объявляю переменные, чтобы не забыть'''
weight = 0  # вес
time = 0  # время
p = 0  # плотность воздуха
a = 0  # ускорение суммарное
th = 0  # сколько пролетел (м)
g = 0  # g
v0 = 0  # начальная скорость
v1 = 0  # конечная скорость
Vseg = 0  # объем сегмента шара
Farh = 0  # сила Архимеда
P = 101325 # давление
R = round(((V * 3) / (math.pi * 4)) ** (1 / 3), 3)
on_hit = True
# delta_v = -1

Fly_arr = [0]  # список веса для графика
a0_arr = []
hit_arr = []
part_water_ar = []
water_ar = []
part_h_arr = [[0], [], [], [], []]

all_weight_arr = [0]
all_h_arr = [0]  # список высот для графика

running_time = False  # идут ли расчеты
check = [0, 0, 0, 0]


'''Создаем начальные элементы: кнопки и текстовые поля'''
Start_b = pgui.elements.UIButton(relative_rect=pg.Rect((50, 10), tp.Start_b_size),
                                 text='Start',
                                 manager=manager)
Restart_b = pgui.elements.UIButton(relative_rect=pg.Rect((260, 10), tp.Restart_b_size),
                                   text='Restart',
                                   manager=manager)
Draw_graph_b = pgui.elements.UIButton(relative_rect=pg.Rect((395, 10), tp.Restart_b_size),
                                      text='Нарисовать',
                                      manager=manager)
Prev_point_b = pgui.elements.UIButton(relative_rect=pg.Rect((530, 10), tp.Restart_b_size),
                                      text='Пред',
                                      manager=manager)
Next_point_b = pgui.elements.UIButton(relative_rect=pg.Rect((665, 10), tp.Restart_b_size),
                                      text='След',
                                      manager=manager)



m_entry = pgui.elements.UITextEntryLine(relative_rect=pg.Rect((tp.Input_x, tp.Input_y), tp.Input_size),
                                        manager=manager)
V_entry = pgui.elements.UITextEntryLine(relative_rect=pg.Rect((tp.Input_x, tp.Input_y + 70), tp.Input_size),
                                        manager=manager)
h_entry = pgui.elements.UITextEntryLine(relative_rect=pg.Rect((tp.Input_x, tp.Input_y + 70 * 2), tp.Input_size),
                                        manager=manager)
h_delta_entry = pgui.elements.UITextEntryLine(relative_rect=pg.Rect((tp.Input_x, tp.Input_y + 70 * 3), tp.Input_size),
                                              manager=manager)

Error_msg = pgui.elements.UILabel(relative_rect=pg.Rect((20, 800), (300, 50)), text='',
                                              manager=manager)

'''Задаем начальный текст'''
m_entry.set_text(str(m))
V_entry.set_text(str(V))
h_entry.set_text(str(h))
h_delta_entry.set_text(str(tp.delta_h))

'''Максимальная длина строки - 10'''
m_entry.set_text_length_limit(tp.limit)
V_entry.set_text_length_limit(tp.limit)
h_entry.set_text_length_limit(tp.limit)
h_delta_entry.set_text_length_limit(tp.limit)

'''Разрешается вписывать только цифры и . , '''
m_entry.set_allowed_characters(tp.white_list)
V_entry.set_allowed_characters(tp.white_list)
h_entry.set_allowed_characters(tp.white_list)
h_delta_entry.set_allowed_characters(tp.white_list)

'''Пишем поясняющие заголовки'''
screen.blit(f1.render(tp.title_m, False, tp.black), (50, 105))
screen.blit(f1.render(tp.title_V, False, tp.black), (50, 175))
screen.blit(f1.render(tp.title_h, False, tp.black), (50, 245))
screen.blit(f1.render(tp.title_delta_h, False, tp.black), (50, 315))
run = True
while run:
    if len(all_h_arr) > 1:
        Draw_graph_b.enable()
    else:
        Draw_graph_b.disable()
    r = round(((V * 3) / (math.pi * 4)) ** (1 / 3), 3)
    S = round(math.pi * (r ** 2), 3)

    time_delta = clock.tick(60) / 1000.0
    for event in pg.event.get():
        if event.type == pg.QUIT:
            run = False

        if event.type == pg.USEREVENT:
            if event.user_type == pgui.UI_BUTTON_PRESSED:
                if event.ui_element == Start_b:  # нажатие на старт
                    if Start_b.text == 'Start':
                        Start_b.set_text('Stop')
                        R = round(((V * 3) / (math.pi * 4)) ** (1 / 3), 3)
                        Error_msg.text = ''
                        running_time = True
                    else:
                        Start_b.set_text('Start')
                        running_time = False
                if event.ui_element == Restart_b:  # нажатие на рестарт
                    Restart_bf()
                if event.ui_element == Draw_graph_b:
                    Draw_graph_bf()

        manager.process_events(event)
    if not running_time:
        enable(True)
        focus_check()
        changes()

    if running_time:
        enable(False)

        if h > 0:  # этап падения
            if 10000 >= h >= tp.delta_h and tp.delta_h > 5:
                tp.delta_h = 5
            if 0 < h <= tp.delta_h and tp.delta_h > 5:
                tp.delta_h = h

            v0 = v1
            g = g_h(h)
            p = p_h(h)
            weight = resist_force(v0, p, S, tp.Cf)
            a = a_F(g, weight)
            try:
                v1 = v_S(a, v0, tp.delta_h)
            except Exception:
                print(v1, v0, g, a, h, p, S, weight)
                Error_msg.set_text('Слишком большой шаг!')
                fig = plt.figure()  # настраиваем размер, чтобы не коверкать картинку
                plt.grid(True)
                plt.plot(part_h_arr[0], Fly_arr, color='b')
                plt.xlabel("Путь")  # подпишем оси
                plt.ylabel("Вес")
                plt.show()
                Restart_bf()
                running_time = False
            print(h, weight, p, v1, a)
            arr_append(th, weight, a, tp.delta_h)
            all_weight_arr.append(weight)
            all_h_arr.append(th + tp.delta_h)
            # weight_delay_control(weight_ar):


            if 10000 < h or tp.delta_h <= 5:
                h = round(h - tp.delta_h, 3)  # тякущая высота
                th = round(th + tp.delta_h, 3)  # сколько пролетел
            if 10000 >= h >= tp.delta_h and tp.delta_h > 5:
                tp.delta_h = 5
                h = round(h - 5, 3)  # тякущая высота
                th = round(th + 5, 3)  # сколько пролетел
            if 0 < h <= tp.delta_h:
                tp.delta_h = h
                th = round(float(h_entry.get_text()), 0)  # сколько пролетел
                h = 0  # тякущая высота

        if h == 0 and len(part_h_arr[2]) == 0:  # удар о воду
            tp.delta_h = float(h_delta_entry.get_text())
            v0 = v1
            weight = resist_force(v1, tp.water_p, S, tp.Cf)
            a = a_F(tp.sea_g, weight)
            print(h, weight, v0, v1, a)
            v1 = 0
            time += tp.delta_t
            hit_arr.append(weight)
            all_weight_arr.append(weight)
            all_h_arr.append(th)
            part_h_arr[2].append(th)

        if -(2 * R) <= h <= 0 and len(part_h_arr[2]) != 0:  # неполное погружение
            # draw_graph()
            print('1', h, weight, v0, v1, a, V, Vseg, R)
            tp.delta_t = 0.005
            v0 = v1
            Vseg = round(Vseg_h(abs(h), R), 5)  # спросить про расчет объема частично погруженного шара
            Farh = Arh_force(Vseg)
            weight = round(m * tp.sea_g - Farh, 4)
            a = round(weight / m, 4)
            v1 = round(v0 + tp.delta_t * a, 4)
            th += round(S_t(v0, tp.delta_t, a), 6)
            h -= round(S_t(v0, tp.delta_t, a), 6)
            print('1', h, weight, v0, v1, a, V, Vseg, R)
            part_water_ar.append(weight)
            all_weight_arr.append(weight)
            all_h_arr.append(th + S_t(v0, tp.delta_t, a))
            part_h_arr[3].append(th + S_t(v0, tp.delta_t, a))

            if weight <= 0:
                running_time = False
                Draw_graph()
                Draw_animation()

        if -10000 <= h < -(2 * R):  # погружение
            print('2', h, weight, v0, v1, a, V, r)
            v0 = v1
            P = P_h(abs(h))
            V = comress_h(abs(h), V)
            Farh = Arh_force(Vseg)
            weight = round(m * tp.sea_g - Farh, 4)
            a = round(weight / m, 4)
            v1 = v_S(a, v0, tp.delta_h)
            h = round(h - tp.delta_h, 2)  # тякущая высота
            th = round(th + tp.delta_h, 2)  # сколько пролетел
            water_ar.append(weight)
            all_weight_arr.append(weight)
            all_h_arr.append(th + tp.delta_h)
            part_h_arr[4].append(th + tp.delta_h)
        if h < -10000:
            running_time = False
            Draw_graph()
            Draw_animation()

        pg.draw.rect(screen, tp.white, (50, tp.text_y, 350, 230))
        draw_text(screen)

    manager.update(time_delta)
    manager.draw_ui(screen)

    clock.tick(tp.FPS)
    pg.display.update()
