import pygame
import pygame_gui as pgui
import Fall_Types as tp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import pandas as pd
import os.path


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
    new_P = P_h(new_h) + 101235
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
    new_P = P_h(new_h) + 101325
    new_V = (old_V * 101325) / (new_P * new_k)
    return round(new_V, 10)


# поиск объема сегмента шара
def Vseg_h(new_h, new_r):
    new_Vseg = math.pi * (new_h ** 2) * (3 * new_r - new_h) / 3
    return new_Vseg


# сила Архимеда
def Arh_force(new_V):
    new_Farh = tp.water_p * tp.sea_g * new_V
    return new_Farh


def water_v(new_m, new_V, new_r):
    new_v = (((new_m * tp.sea_g - tp.water_p * tp.sea_g * new_V) * 2) / (tp.Cf * tp.water_p * math.pi * new_r ** 2)) ** (1 / 2)
    return round(new_v, 4)


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
    global start_V, start_h, start_m, check, m_entry, V_entry, h_entry, h, m, V, start_V, h_delta_entry, fly_arr_w, a0_arr_w, hit_arr_w, all_weight_arr, part_h_arr, part_water_ar_w, water_ar_w, all_h_arr, th, fly_arr_v, a0_arr_v, hit_arr_v, part_water_ar_v, water_ar_v, all_v_arr
    if not m_entry.is_focused and check[0] == 1:
        m_entry.set_text(limit_check(m_entry.get_text(), '1', '1', '1')[0])
        m = float(m_entry.get_text())
        start_m = m
        check[0] = 0

    if not V_entry.is_focused and check[1] == 1:
        V_entry.set_text(limit_check('1', V_entry.get_text(), '1', '1')[1])
        V = float(V_entry.get_text())
        start_V = V
        check[1] = 0
    if not h_entry.is_focused and check[2] == 1:
        h_entry.set_text(limit_check('1', '1', h_entry.get_text(), '1')[2])
        h = float(h_entry.get_text())
        start_h = h
        check[2] = 0
        fly_arr_w = [0]  # список веса для графика
        a0_arr_w = []
        hit_arr_w = []
        part_water_ar_w = []
        water_ar_w = []
        fly_arr_v = [0]  # список скоростей для графика
        a0_arr_v = []
        hit_arr_v = []
        part_water_ar_v = []
        water_ar_v = []
        part_h_arr = [[0], [], [], [], []]
        all_v_arr = [0]
        all_weight_arr = [0]
        all_h_arr = [0]  # список высот для графика
        th = 0
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
def enable_text_field(yes):
    global m_entry, V_entry, h_entry
    if yes:
        m_entry.enable()
        V_entry.enable()
        h_entry.enable()

        h_delta_entry.disable()  # УБРАТЬ НА СОВСЕМ
    if not yes:
        m_entry.disable()
        V_entry.disable()
        h_entry.disable()
        h_delta_entry.disable()


# отрисовка текста
def draw_text(window):
    global out_font
    screen.blit(out_font.render('Ускорение    ' + str(round(a, 3)), False, tp.black), (50, tp.text_y))
    screen.blit(out_font.render('Вес  ' + str(round(new_all_weight_arr[-1], 3)), False, tp.black), (50, tp.text_y + 70))
    screen.blit(out_font.render('Скорось  ' + str(round(new_all_v_arr[-1], 3)), False, tp.black), (50, tp.text_y + 70 * 2))
    screen.blit(out_font.render('Высота   ' + str(round(start_h - new_all_h_arr[-1], 3)), False, tp.black), (50, tp.text_y + 70 * 3))

    window.blit(out_font.render('м/с2', False, tp.black), (tp.text_x, tp.text_y))
    window.blit(out_font.render('Н', False, tp.black), (tp.text_x, tp.text_y + 70))
    window.blit(out_font.render('м/с', False, tp.black), (tp.text_x, tp.text_y + 70 * 2))
    window.blit(out_font.render('м', False, tp.black), (tp.text_x, tp.text_y + 70 * 3))


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
    global fly_arr_w, a0_arr_w, hit_arr_w, part_h_arr, h_entry, v1, a0_arr_v

    if round(new_a, 2) == 0:  # or len(a0_arr_w) > 0:
        a0_arr_w.append(new_weight)
        a0_arr_v.append(v1)
        part_h_arr[1].append(new_th + new_delta_h)
    else:
        fly_arr_w.append(new_weight)
        fly_arr_v.append(v1)
        part_h_arr[0].append(new_th + new_delta_h)
        part_h_arr[1] = []
        a0_arr_w = []
        a0_arr_v = []


def Draw_graph(vertically, show, save):
    fig = plt.figure(figsize=(6.4 * 2.4, 4.8 * 1.2))
    plt.subplot(1, 2, 1)

    all_max = []
    if len(fly_arr_w) > 0:
        all_max.append(max(fly_arr_w))
    if len(a0_arr_w) > 0:
        all_max.append(max(a0_arr_w))
    if len(part_water_ar_w) > 0:
        all_max.append(max(part_water_ar_w))
    if len(water_ar_w) > 0:
        all_max.append(max(water_ar_w))

    if len(all_max) > 0:
        max_y = max(all_max)
    elif len(hit_arr_w) > 0:
        max_y = max(hit_arr_w)
    else:
        print('ОШИБКА')


    plt.ylim(0, max_y * 1.2)
    plt.grid(True)

    if len(part_h_arr[0]) > 0:
        plt.plot(part_h_arr[0], fly_arr_w, color='b', label='Полет с ускорением')

    if len(part_h_arr[1]) > 0:
        plt.plot(part_h_arr[1], a0_arr_w, color='g')
        if len(part_h_arr[0]) > 0:
            plt.plot([part_h_arr[0][-1], part_h_arr[1][0]], [fly_arr_w[-1], a0_arr_w[0]], color='g',
                     label='Полет без ускорения')

    if len(part_h_arr[1]) == 0 and len(part_h_arr[2]) > 0 and len(part_h_arr[0]) > 0:
        plt.plot([part_h_arr[0][-1], part_h_arr[2][0]], [fly_arr_w[-1], hit_arr_w[0]], color='r')

    if len(part_h_arr[2]) > 0:
        plt.plot(part_h_arr[2], hit_arr_w, color='r')
        if len(part_h_arr[1]) > 0:
            plt.plot([part_h_arr[1][-1], part_h_arr[2][0]], [a0_arr_w[-1], hit_arr_w[0]], color='r')
        if len(part_h_arr[3]) > 0:
            plt.text(part_h_arr[3][0], 110, f' Fудара = {round(hit_arr_w[-1])} Н',
                     bbox=tp.box)
            plt.plot([part_h_arr[2][-1], part_h_arr[3][0]], [hit_arr_w[-1], part_water_ar_w[0]], color='r',
                     label='удар о воду')


    if len(part_h_arr[3]) > 0:
        plt.plot(part_h_arr[3], part_water_ar_w, color='c', label='Неполное погружение')
        if len(part_h_arr[4]) > 0:
            plt.plot([part_h_arr[3][-1], part_h_arr[4][0]], [part_water_ar_w[-1], water_ar_w[0]], color='m',
                 label='погружение')

    if len(part_h_arr[4]) > 0:
        plt.plot(part_h_arr[4], water_ar_w, color='m')


    plt.legend()
    plt.xlabel("Высота(м)")  # подпишем оси
    plt.ylabel("Вес(Н)")

    h = float(h_entry.get_text())
    locs, labels = plt.xticks()
    plt.xticks(ticks=locs, labels=[round(h - i) for i in plt.xticks()[0]])
    plt.ylim(0, max_y * 1.2)

    plt.subplot(1, 2, 2)
    plt.grid(True)
    if len(part_h_arr[0]) > 0:
        plt.plot(part_h_arr[0], fly_arr_v, color='b', label='Полет с ускорением')

    if len(part_h_arr[1]) > 0 and len(part_h_arr[0]) > 0:
        plt.plot([part_h_arr[0][-1], part_h_arr[1][0]], [fly_arr_v[-1], a0_arr_v[0]], color='g',
                 label='Полет без ускорения')
        plt.plot(part_h_arr[1], a0_arr_v, color='g')
        if len(part_h_arr[2]) > 0:
            plt.plot([part_h_arr[1][-1], part_h_arr[2][0]], [a0_arr_v[-1], hit_arr_v[0]], color='r')

    if len(part_h_arr[1]) == 0 and len(part_h_arr[2]) > 0:
        plt.plot([part_h_arr[0][-1], part_h_arr[2][0]], [fly_arr_v[-1], hit_arr_v[0]], color='r')

    if len(part_h_arr[2]) > 0:
        plt.plot(part_h_arr[2], hit_arr_v, color='r')
        if len(part_h_arr[3]) > 0:
            plt.plot([part_h_arr[2][-1], part_h_arr[3][0]], [hit_arr_v[-1], part_water_ar_v[0]], color='r',
                     label='удар о воду')

    if len(part_h_arr[3]) > 0:
        plt.plot(part_h_arr[3], part_water_ar_v, color='c', label='Неполное погружение')

    if len(part_h_arr[4]) > 0 and len(part_h_arr[3]) > 0:
        plt.plot([part_h_arr[3][-1], part_h_arr[4][0]], [part_water_ar_v[-1], water_ar_v[0]], color='m',
                 label='погружение')
        plt.plot(part_h_arr[4], water_ar_v, color='m')

    plt.legend()
    plt.xlabel("Высота(м)")  # подпишем оси
    plt.ylabel("Скорость(м/с2)")

    locs, labels = plt.xticks()
    plt.xticks(ticks=locs, labels=[round(h - i) for i in plt.xticks()[0]])

    if save:
        i = 1
        while True:
            if os.path.isfile(f'График {i}.png'):
                i += 1
            else:
                plt.savefig(f'График {i}.png')
                break
    if show:
        plt.show()


def Draw_w_animation(vertically):
    fig, ax = plt.subplots(figsize=(6.4 * 1.7, 4.8 * 1.7))
    ax = plt.axis([0, max(all_h_arr) * 1.1, 0, tp.max_m * 12])
    plt.xlabel('Путь')
    plt.ylabel('Вес')
    red_dot, = plt.plot([0], [0], 'ro')
    print(len(all_weight_arr), len(all_h_arr))
    plt.grid(True)

    def animate(i):
        red_dot.set_data(all_h_arr[i], all_weight_arr[i])
        line = plt.plot(all_h_arr[:i], all_weight_arr[:i], 'r')
        return red_dot,

    if len(all_h_arr) < 1500:
        ani = FuncAnimation(fig, animate, interval=10, repeat=True, frames=len(all_weight_arr))
        i = 1
        while True:
            if os.path.isfile(f'График {i}.png'):
                i += 1
            else:
                ani.save(f'Анимация веса {i}.gif', writer="pillow")
                break

    plt.close(fig)


def Draw_v_animation(vertically):
    fig, ax = plt.subplots(figsize=(6.4 * 1.7, 4.8 * 1.7))
    plt.xlabel('Путь')
    plt.ylabel('Вес')
    red_dot, = plt.plot([0], [0], 'ro')
    print(len(all_v_arr), len(all_h_arr))
    plt.grid(True)

    def animate(i):
        red_dot.set_data(all_h_arr[i], all_v_arr[i])
        line = plt.plot(all_h_arr[:i], all_v_arr[:i], 'b')
        return red_dot,

    if len(all_h_arr) < 1500:
        ani = FuncAnimation(fig, animate, interval=10, repeat=True, frames=len(all_v_arr))
        i = 1
        while True:
            if os.path.isfile(f'График {i}.png'):
                i += 1
            else:
                ani.save(f'Анимация скорости {i}.gif', writer="pillow")
                break

    plt.close(fig)



def Reset_Variable(reset_type):
    global weight, time, p, a, th, g, v0, v1, Vseg, Farh, P, r, all_weight_arr, part_h_arr, all_h_arr, fly_arr_w, a0_arr_w, hit_arr_w, part_water_ar_w, water_ar_w, fly_arr_v, a0_arr_v, hit_arr_v, part_water_ar_v, water_ar_v, all_v_arr
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
    r = round(((V * 3) / (math.pi * 4)) ** (1 / 3), 3)
    P = 101325  # давление

    a0_arr_w = []
    hit_arr_w = []
    all_weight_arr = []
    part_water_ar_w = []
    water_ar_w = []

    a0_arr_v = []
    hit_arr_v = []
    part_water_ar_v = []
    water_ar_v = []

    if reset_type == 1:
        fly_arr_w = [0]  # список веса для графика
        part_h_arr = [[0], [], [], [], []]
        all_weight_arr = [0]
        all_v_arr = [0]
        all_h_arr = [0]  # список высот для графика
        fly_arr_v = [0]  # список скоростей для графика

    if reset_type == 2:
        fly_arr_w = []  # список веса для графика
        part_h_arr = [[], [], [], [], []]
        all_weight_arr = []
        all_v_arr = []
        all_h_arr = []  # список высот для графика
        fly_arr_v = []  # список скоростей для графика

def Correct_delta_h(new_h):
    if 0 < new_h <= 1:
        new_delta_h = new_h
    elif new_h <= 0:
        if 0 >= new_h > -50:
            new_delta_h = 1
        elif -50 >= new_h > 300:
            new_delta_h = 3
        else:
            new_delta_h = 3
    else:
        if new_h // 2500 >= 1:
            new_delta_h = new_h // 2500
        else:
            new_delta_h = 1

    return new_delta_h


def Start_bf():
    global r, running_time
    if Start_b.text == 'Start':
        Start_b.set_text('Stop')
        # r = round(((V * 3) / (math.pi * 4)) ** (1 / 3), 3)
        Error_msg.text = ''
        running_time = True
    else:
        Start_b.set_text('Start')
        running_time = False


def Restart_bf():
    global running_time, m, V, h, r, start_V, end_of_simulation, prev_point
    Start_b.set_text('Start')
    running_time = False
    end_of_simulation = False
    prev_point = False

    running_time = False  # идут ли расчеты
    Reset_Variable(1)
    m = float(m_entry.get_text())
    V = float(V_entry.get_text())
    start_V = V
    h = float(h_entry.get_text())
    r = round(((V * 3) / (math.pi * 4)) ** (1 / 3), 3)
    tp.delta_h = Correct_delta_h(h)  # функция определения шага от высоты
    pygame.draw.rect(screen, tp.white, (50, tp.text_y, 250, 230))
    draw_text(screen)


def Draw_graph_bf():
    global running_time, m, V, h, r
    running_time = False
    Start_b.disable()
    Restart_b.disable()
    m_entry.disable()
    V_entry.disable()
    h_entry.disable()
    h_delta_entry.disable()

    Start_b.set_text('Start')
    Draw_graph(vertically, True, False)
    # Draw_animation()

    Start_b.enable()
    Restart_b.enable()
    m_entry.enable()
    V_entry.enable()
    h_entry.enable()

    h_delta_entry.disable()


def Graph_type_bf():
    global graph_type
    if Graph_type_b.text == 'P(h)':
        Graph_type_b.set_text('v(h)')
        graph_type = 2
        return None
    if Graph_type_b.text == 'v(h)':
        Graph_type_b.set_text('P(h)')
        graph_type = 1
        return None


def Save_animation_bf():
    global saving_anim
    if Save_animation_b.text == 'Сохр. аним.':
        Save_animation_b.set_text('Не сохр. аним.')
        saving_anim = False
        return None
    if Save_animation_b.text == 'Не сохр. аним.':
        Save_animation_b.set_text('Сохр. аним.')
        saving_anim = True
        return None


def Graph_reverse_bf():
    global vertically
    if Graph_reverse_b.text == 'вертикально':
        Graph_reverse_b.set_text('горизонтально')
        vertically = False
        return None
    if Graph_reverse_b.text == 'горизонтально':
        Graph_reverse_b.set_text('вертикально')
        vertically = True
        return None


def Prev_point_bf(all_h, all_weight, all_v):
    global running_time, prev_point
    running_time = False
    Start_b.set_text('Start')
    new_all_h = all_h[:-1]
    new_all_weight = all_weight[:-1]
    new_all_v = all_v[:-1]
    prev_point = True
    return new_all_h, new_all_weight, new_all_v


def Next_point_bf(all_h, all_weight, all_v):
    global running_time, prev_point
    running_time = False
    Start_b.set_text('Start')
    new_all_h = all_h_arr[:len(all_h) + 1]
    new_all_weight = all_weight_arr[:len(all_weight) + 1]
    new_all_v = all_v_arr[:len(all_v) + 1]
    if len(all_h_arr) == len(new_all_h):
        prev_point = False
    return new_all_h, new_all_weight, new_all_v


def Save_csv():
    global all_v_arr, all_h_arr, all_weight_arr
    all_h = [round(i, 2) for i in all_h_arr]
    all_weight = [round(i, 2) for i in all_weight_arr]
    all_v = [round(i, 2) for i in all_v_arr]
    dic = {'way': all_h,
           'weight': all_weight,
           'speed': all_v}

    i = 1
    while True:
        if os.path.isfile(f'График {i}.png'):
            i += 1
        else:
            name = f'Данные {i}.csv'
            break

    df = pd.DataFrame(dic, columns=['way', 'weight', 'speed'])
    df.to_csv(path_or_buf=name, index=False, sep='\t')


def Save_all_bf():
    global running_time, vertically, all_v_arr, all_h_arr, all_weight_arr, saving_anim
    running_time = False
    Start_b.set_text('Start')
    if saving_anim:
        Draw_w_animation(vertically)
        Draw_v_animation(vertically)
    Save_csv()
    Draw_graph(vertically, True, True)

def correct_borders(max_x, max_y):
    if max_x < 1000:
        new_max_x = 100 * (max_x // 100 + 1)
    else:
        new_max_x = 1000 * (max_x // 1000 + 1)
    if max_y < 1000:
        new_max_y = 100 * (max_y // 100 + 1)
    else:
        new_max_y = 1000 * (max_y // 1000 + 1)
    return new_max_x, new_max_y


class Axis:
    def __init__(self, ab_x, ab_y, ord_x, ord_y, ab_len, ord_len):
        """
        :param ab_x: начало оси абсцисс по х
        :param ab_y: начало оси абсцисс по y
        :param ord_x: начало оси ординат по х
        :param ord_y: начало оси ординат по y
        :param ab_len: длина оси абсцисс
        :param ord_len: длина оси ординат
        """

        self.ab_x = ab_x
        self.ord_x = ord_x
        self.ab_y = ab_y
        self.ord_y = ord_y
        self.ab_len = ab_len
        self.ord_len = ord_len
        self.start_pos_ab = (ab_x, ab_y)
        self.start_pos_ord = (ord_x, ord_y)
        self.end_pos_ab = (ab_x + ab_len, ab_y)
        self.end_pos_ord = (ord_x, ord_y - ord_len)

    def draw(self, surface):
        pygame.draw.line(surface, tp.black, self.start_pos_ab, self.end_pos_ab, width=3)
        pygame.draw.line(surface, tp.black, self.start_pos_ord, self.end_pos_ord, width=3)
        pygame.draw.polygon(surface, tp.black,
                            [(self.ab_x + self.ab_len + 7, self.ab_y), (self.ab_x + self.ab_len, self.ab_y + 4),
                             (self.ab_x + self.ab_len, self.ab_y - 4)])
        pygame.draw.polygon(surface, tp.black,
                            [(self.ord_x, self.ord_y - self.ord_len - 7), (self.ord_x - 4, self.ord_y - self.ord_len),
                             (self.ord_x + 4, self.ord_y - self.ord_len)])

        for i in range(self.ab_len // tp.ab_interval + 1):
            pygame.draw.line(surface, tp.black, (self.ab_x + i * tp.ab_interval, self.ab_y - 5),
                             (self.ab_x + i * tp.ab_interval, self.ab_y + 5), width=3)
            if i != 0:
                pygame.draw.line(surface, (220, 220, 220), (self.ab_x + i * tp.ab_interval, self.ab_y - 5),
                                 (self.ab_x + i * tp.ab_interval, self.ab_y - self.ord_len), width=3)
        for i in range(self.ord_len // tp.ord_interval + 1):
            pygame.draw.line(surface, tp.black, (self.ord_x - 5, self.ord_y - i * tp.ord_interval),
                             (self.ord_x + 5, self.ord_y - i * tp.ord_interval), width=3)
            if i != 0:
                pygame.draw.line(surface, (220, 220, 220), (self.ord_x + 5, self.ord_y - i * tp.ord_interval),
                                 (self.ord_x + self.ab_len, self.ord_y - i * tp.ord_interval), width=3)


    def sign_devision(self, surface, max_x, max_y, vertically):
        h0 = float(h_entry.get_text())
        if vertically:
            max_x, max_y = correct_borders(max_x, max_y)

            for i in range(self.ab_len // tp.ab_interval + 1):
                text = h0 - max_x / self.ab_len * tp.ab_interval * i
                surface.blit(graph_font.render(f'{round(text)}', False, tp.black),
                             (self.ab_x + i * tp.ab_interval, self.ab_y + 5))

            for i in range(self.ord_len // tp.ord_interval + 1):
                text = max_y / self.ord_len * tp.ord_interval * i
                surface.blit(graph_font.render(f'{round(text)}', False, tp.black),
                             (self.ord_x - len(str(round(text))) * 12, self.ord_y - i * tp.ord_interval))
        else:
            max_y, max_x = correct_borders(max_x, max_y)

            for i in range(self.ab_len // tp.ab_interval + 1):
                text = max_x / self.ab_len * tp.ab_interval * i
                surface.blit(graph_font.render(f'{round(text)}', False, tp.black),
                             (self.ab_x + i * tp.ab_interval, self.ab_y + 5))

            for i in range(self.ord_len // tp.ord_interval + 1):
                text = h0 - (max_y - max_y / self.ord_len * tp.ord_interval * i)
                surface.blit(graph_font.render(f'{round(text)}', False, tp.black),
                             (self.ord_x - len(str(round(text))) * 12, self.ord_y - i * tp.ord_interval))


    def draw_weight_graph(self, surface, all_h, all_weight, max_x, max_y, vertically):
        if vertically:
            max_x, max_y = correct_borders(max_x, max_y)
            surface.blit(graph_font.render('высота', False, tp.black), (self.ab_x + self.ab_len + 10, self.ab_y - 25))
            surface.blit(graph_font.render('вес', False, tp.black), (self.ord_x - 1.5 * 10, self.ord_y - self.ord_len - 35))
            x = [i * (self.ab_len / max_x) for i in all_h]
            y = [i * (self.ord_len / max_y) for i in all_weight]
            for i in range(len(all_h) - 1):
                pygame.draw.line(surface, tp.red, (x[i] + self.ab_x, self.ab_y - y[i]),
                                 (x[i + 1] + self.ab_x, self.ab_y - y[i + 1]), width=3)
        else:
            max_y, max_x = correct_borders(max_x, max_y)
            surface.blit(graph_font.render('вес', False, tp.black), (self.ab_x + self.ab_len + 10, self.ab_y - 25))
            surface.blit(graph_font.render('высота', False, tp.black), (self.ord_x - 3 * 10, self.ord_y - self.ord_len - 35))
            x = [i * (self.ab_len / max_x) for i in all_weight]
            y = [self.ord_len - i * (self.ord_len / max_y) for i in all_h]
            for i in range(len(all_h) - 1):
                pygame.draw.line(surface, tp.red, (x[i] + self.ab_x, self.ab_y - y[i]),
                                 (x[i + 1] + self.ab_x, self.ab_y - y[i + 1]), width=3)


    def draw_v_graph(self, surface, all_h, all_v, max_x, max_y, vertically):
        if vertically:
            max_x, max_y = correct_borders(max_x, max_y)
            surface.blit(graph_font.render('высота', False, tp.black), (self.ab_x + self.ab_len + 10, self.ab_y - 25))
            surface.blit(graph_font.render('скорость', False, tp.black), (self.ord_x - 4 * 10, self.ord_y - self.ord_len - 35))
            x = [i * (self.ab_len / max_x) for i in all_h]
            y = [i * (self.ord_len / max_y) for i in all_v]
            for i in range(len(all_h) - 1):
                pygame.draw.line(surface, tp.blue, (x[i] + self.ab_x, self.ab_y - y[i]),
                                 (x[i + 1] + self.ab_x, self.ab_y - y[i + 1]), width=3)
        else:
            max_y, max_x = correct_borders(max_x, max_y)
            surface.blit(graph_font.render('скорость', False, tp.black), (self.ab_x + self.ab_len + 10, self.ab_y - 25))
            surface.blit(graph_font.render('высота', False, tp.black), (self.ord_x - 3 * 10, self.ord_y - self.ord_len - 35))
            x = [i * (self.ab_len / max_x) for i in all_v]
            y = [i * (self.ord_len / max_y) for i in all_h]
            for i in range(len(all_h) - 1):
                pygame.draw.line(surface, tp.blue, (x[i] + self.ab_x, self.ab_y - y[i]),
                                 (x[i + 1] + self.ab_x, self.ab_y - y[i + 1]), width=3)


'''Создаем окно, настраиваем шрифт и часы'''
pygame.init()
pygame.font.init()
out_font = pygame.font.Font(None, 30)
graph_font = pygame.font.Font(None, 25)
manager = pgui.UIManager((tp.WIDTH, tp.HEIGHT))
screen = pygame.display.set_mode((tp.WIDTH, tp.HEIGHT))
pygame.display.set_caption("Falling Ball")
clock = pygame.time.Clock()
screen.fill(tp.white)

'''Предварительно задаем переменные'''
m = 10
V = 10
h = 200000
start_h = 0
start_V = 10
start_m = 10

new_start_h = 0
new_start_v = 0
new_start_weight = 0

r = ((V * 3) / (math.pi * 4)) ** (1 / 3)
S = math.pi * r ** 2

'''Объявляю переменные'''
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
P = 101325  # давление



fly_arr_w = [0]  # список веса для графика
a0_arr_w = []
hit_arr_w = []
part_water_ar_w = []
water_ar_w = []
fly_arr_v = [0]  # список веса для графика
a0_arr_v = []
hit_arr_v = []
part_water_ar_v = []
water_ar_v = []
part_h_arr = [[0], [], [], [], []]

all_v_arr = [0]
all_weight_arr = [0]
all_h_arr = [0]  # список высот для графика

new_all_v_arr = all_v_arr
new_all_weight_arr = all_weight_arr
new_all_h_arr = all_h_arr

end_of_simulation = False
vertically = True
graph_type = 1
prev_point = False
saving_anim = True
running_time = False  # идут ли расчеты
check = [0, 0, 0, 0]

'''Создаем начальные элементы: кнопки и текстовые поля'''
Start_b = pgui.elements.UIButton(relative_rect=pygame.Rect((50, 10), tp.Start_b_size),
                                 text='Start',
                                 manager=manager)
Restart_b = pgui.elements.UIButton(relative_rect=pygame.Rect((260, 10), tp.Restart_b_size),
                                   text='Restart',
                                   manager=manager)
Draw_graph_b = pgui.elements.UIButton(relative_rect=pygame.Rect((395, 10), tp.Restart_b_size),
                                      text='Нарисовать',
                                      manager=manager)
Prev_point_b = pgui.elements.UIButton(relative_rect=pygame.Rect((530, 10), tp.Restart_b_size),
                                      text='Пред.',
                                      manager=manager)
Next_point_b = pgui.elements.UIButton(relative_rect=pygame.Rect((665, 10), tp.Restart_b_size),
                                      text='След.',
                                      manager=manager)
Graph_type_b = pgui.elements.UIButton(relative_rect=pygame.Rect((800, 10), tp.Restart_b_size),
                                      text='P(h)',
                                      manager=manager)
Graph_reverse_b = pgui.elements.UIButton(relative_rect=pygame.Rect((935, 10), tp.Restart_b_size),
                                         text='вертикально',
                                         manager=manager)
Save_animation_b = pgui.elements.UIButton(relative_rect=pygame.Rect((1070, 10), tp.Restart_b_size),
                                          text='Сохр. аним.',
                                          manager=manager)
Save_all_b = pgui.elements.UIButton(relative_rect=pygame.Rect((1205, 10), tp.Restart_b_size),
                                    text='Сохр. все',
                                    manager=manager)

m_entry = pgui.elements.UITextEntryLine(relative_rect=pygame.Rect((tp.Input_x, tp.Input_y), tp.Input_size),
                                        manager=manager)
V_entry = pgui.elements.UITextEntryLine(relative_rect=pygame.Rect((tp.Input_x, tp.Input_y + 70), tp.Input_size),
                                        manager=manager)
h_entry = pgui.elements.UITextEntryLine(relative_rect=pygame.Rect((tp.Input_x, tp.Input_y + 70 * 2), tp.Input_size),
                                        manager=manager)
h_delta_entry = pgui.elements.UITextEntryLine(
    relative_rect=pygame.Rect((tp.Input_x, tp.Input_y + 70 * 3), tp.Input_size),
    manager=manager)

Error_msg = pgui.elements.UILabel(relative_rect=pygame.Rect((20, 800), (250, 50)), text='',
                                  manager=manager)

surface1 = pygame.Surface((1250, 800), flags=0)
surface1.fill((240, 240, 240))
axis1 = Axis(100, 700, 100, 700, 1000, 600)
axis2 = Axis(100, 700, 100, 700, 1000, 600)
# pygame.draw.line(surface1, tp.black, (625, 0), (625, 800))
# pygame.draw.line(surface1, tp.black, (0, 400), (1250, 0))
axis1.draw(surface1)

'''Задаем начальный текст'''
m_entry.set_text(str(m))
V_entry.set_text(str(V))
h_entry.set_text(str(h))
h_delta_entry.set_text('~')

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
screen.blit(out_font.render(tp.title_m, False, tp.black), (50, 105))
screen.blit(out_font.render(tp.title_V, False, tp.black), (50, 175))
screen.blit(out_font.render(tp.title_h, False, tp.black), (50, 245))
screen.blit(out_font.render(tp.title_delta_h, False, tp.black), (50, 315))

screen.blit(graph_font.render(f'{tp.min_m}-{tp.max_m}', False, (30, 30, 30)), (tp.Input_x, tp.Input_y + 30))
screen.blit(graph_font.render(f'{tp.min_v}-{tp.max_v}', False, (30, 30, 30)), (tp.Input_x, tp.Input_y + 100))
screen.blit(graph_font.render(f'{tp.min_h}-{tp.max_h}', False, (30, 30, 30)), (tp.Input_x, tp.Input_y + 170))
screen.blit(out_font.render('~', False, (30, 30, 30)), (tp.Input_x + 5, tp.Input_y + 240))

run = True
while run:
    if len(all_h_arr) > 1:
        Draw_graph_b.enable()
        Save_all_b.enable()
    else:
        Draw_graph_b.disable()
        Save_all_b.disable()

    if len(new_all_h_arr) > 1:
        Prev_point_b.enable()
    else:
        Prev_point_b.disable()

    if prev_point:
        Next_point_b.enable()
    else:
        Next_point_b.disable()


    time_delta = clock.tick(60) / 1000.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            pygame.quit()


        if event.type == pygame.USEREVENT:
            if event.user_type == pgui.UI_BUTTON_PRESSED:  # обработка нажатий на кнопки
                if event.ui_element == Start_b:  # нажатие на старт
                    if end_of_simulation:
                        Restart_bf()
                        end_of_simulation = False
                    elif prev_point:
                        Save_all_bf()
                        Reset_Variable(2)
                        th = new_all_h_arr[-1]
                        h = start_h - th
                        v1 = new_all_v_arr[-1]
                        weight = new_all_weight_arr[-1]
                        prev_point = False
                        Start_bf()
                    else:
                        Start_bf()
                if event.ui_element == Restart_b:  # нажатие на рестарт
                    Restart_bf()
                if event.ui_element == Draw_graph_b:
                    Draw_graph_bf()
                if event.ui_element == Graph_type_b:
                    Graph_type_bf()
                if event.ui_element == Save_animation_b:
                    Save_animation_bf()
                if event.ui_element == Next_point_b:
                    new_all_h_arr, new_all_weight_arr, new_all_v_arr = Next_point_bf(new_all_h_arr, new_all_weight_arr, new_all_v_arr)
                if event.ui_element == Prev_point_b:
                    new_all_h_arr, new_all_weight_arr, new_all_v_arr = Prev_point_bf(new_all_h_arr, new_all_weight_arr, new_all_v_arr)
                if event.ui_element == Graph_reverse_b:
                    Graph_reverse_bf()
                if event.ui_element == Save_all_b:
                    Save_all_bf()

        manager.process_events(event)
    if not running_time:
        enable_text_field(True)
        focus_check()
        changes()

    if running_time:

        new_all_v_arr = all_v_arr
        new_all_weight_arr = all_weight_arr
        new_all_h_arr = all_h_arr

        enable_text_field(False)

        if h > 0:  # этап падения
            r = round(((start_V * 3) / (math.pi * 4)) ** (1 / 3), 3)
            S = round(math.pi * (r ** 2), 3)
            tp.delta_h = Correct_delta_h(h)
            v0 = v1
            g = g_h(h)
            p = p_h(h)
            weight = resist_force(v0, p, S, tp.Cf)
            a = a_F(g, weight)

            try:
                v1 = v_S(a, v0, tp.delta_h)
            except:
                print(v1, v0, g, a, h, p, S, weight)
                Restart_bf()
                running_time = False
                Start_b.set_text('Start')

            arr_append(th, weight, a, tp.delta_h)
            all_weight_arr.append(weight)
            all_v_arr.append(v1)
            all_h_arr.append(round(th + tp.delta_h, 6))

            if 200000 >= h > 1:
                h = round(h - tp.delta_h, 3)  # тякущая высота
                th = round(th + tp.delta_h, 3)  # сколько пролетел
            if 0 < h <= 1:
                th = round(float(h_entry.get_text()), 0)  # сколько пролетел
                h = 0  # тякущая высота


        if h == 0 and len(part_h_arr[2]) == 0:  # удар о воду
            tp.delta_h = 10
            v0 = v1
            weight = resist_force(v1, tp.water_p, S, 1)
            a = a_F(tp.sea_g, weight)
            v1 = 0

            time += tp.delta_t

            hit_arr_w.append(weight)
            hit_arr_v.append(v1)
            all_weight_arr.append(weight)
            all_v_arr.append(v1)
            all_h_arr.append(th)
            part_h_arr[2].append(th)

        if -(2 * r) <= h <= 0 and len(part_h_arr[2]) != 0:  # неполное погружение
            tp.delta_t = 0.005
            v0 = v1
            Vseg = round(Vseg_h(abs(h), r), 5)  # спросить про расчет объема частично погруженного шара
            Farh = Arh_force(Vseg)
            weight = round(m * tp.sea_g - Farh, 4)
            a = round(weight / m, 4)
            v1 = round(v0 + tp.delta_t * a, 4)

            th += round(S_t(v0, tp.delta_t, a), 6)
            h -= round(S_t(v0, tp.delta_t, a), 6)

            part_water_ar_w.append(weight)
            part_water_ar_v.append(v1)
            all_weight_arr.append(weight)
            all_h_arr.append(th + S_t(v0, tp.delta_t, a))
            all_v_arr.append(v1)
            part_h_arr[3].append(th + S_t(v0, tp.delta_t, a))
            tp.depth = float(h_entry.get_text()) * 2  # ДЛЯ ТЕСТОВ!!!

            if weight <= 0:
                part_water_ar_w[-1] = 0
                all_weight_arr[-1] = 0
                Start_bf()
                if saving_anim:
                    Draw_w_animation(vertically)
                    Draw_v_animation(vertically)
                Save_csv()
                Draw_graph(vertically, True, True)
                end_of_simulation = True

        if -tp.depth <= h < -(2 * r):  # погружение
            tp.delta_h = Correct_delta_h(h)

            if tp.delta_h > tp.depth + h:
                tp.delta_h = tp.depth + h

            v0 = v1
            if len(water_ar_w) > 0:
                V = comress_h(abs(h), start_V)  # ???
            else:
                V = Vseg

            r = round(((V * 3) / (math.pi * 4)) ** (1 / 3), 5)
            S = round(math.pi * (r ** 2), 3)
            Farh = Arh_force(V)
            weight = round(m * tp.sea_g - Farh, 10)

            v1 = water_v(m, V, r)
            print(h, V, r, v1, k_h(abs(h)))
            a = (v1 ** 2 - v0 ** 2) / (2 * tp.delta_h)
            h = round(h - tp.delta_h, 2)  # тякущая высота
            th = round(th + tp.delta_h, 2)  # сколько пролетел

            water_ar_w.append(weight)
            water_ar_v.append(v1)
            all_weight_arr.append(weight)
            all_v_arr.append(v1)
            all_h_arr.append(th + tp.delta_h)
            part_h_arr[4].append(th + tp.delta_h)

        if h <= -tp.depth:
            Start_bf()
            if saving_anim:
                Draw_w_animation(vertically)
                Draw_v_animation(vertically)
            Save_csv()
            Draw_graph(vertically, True, True)
            end_of_simulation = True

    pygame.draw.rect(screen, tp.white, (50, tp.text_y, 250, 230))
    draw_text(screen)

    surface1.fill((240, 240, 240))
    axis1.draw(surface1)
    if len(all_h_arr) > 1:
        if not prev_point:
            if graph_type == 1:
                axis1.sign_devision(surface1, max(all_h_arr), max(all_weight_arr), vertically)
                axis1.draw_weight_graph(surface1, all_h_arr, all_weight_arr, max(all_h_arr), max(all_weight_arr),
                                        vertically)
            if graph_type == 2:
                axis1.sign_devision(surface1, max(all_h_arr), max(all_v_arr), vertically)
                axis1.draw_v_graph(surface1, all_h_arr, all_v_arr, max(all_h_arr), max(all_v_arr), vertically)
        if prev_point:
            if graph_type == 1:
                axis1.sign_devision(surface1, max(new_all_h_arr), max(new_all_weight_arr), vertically)
                axis1.draw_weight_graph(surface1, new_all_h_arr, new_all_weight_arr, max(new_all_h_arr), max(new_all_weight_arr),
                                        vertically)
            if graph_type == 2:
                axis1.sign_devision(surface1, max(new_all_h_arr), max(new_all_v_arr), vertically)
                axis1.draw_v_graph(surface1, new_all_h_arr, new_all_v_arr, max(new_all_h_arr), max(new_all_v_arr), vertically)

    manager.update(time_delta)
    manager.draw_ui(screen)

    screen.blit(surface1, (350, 100))
    clock.tick(tp.FPS)
    pygame.display.flip()
