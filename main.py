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
def comress_h(new_h, old_V, new_r):
    new_k = k_h(new_h)
    new_P = P_h(new_h) + 101325
    new_V = (old_V * 101325) / new_P
    if new_V > old_V:
        new_V = old_V
    return new_V


# поиск объема сегмента шара
def Vseg_h(new_h, new_r):
    new_Vseg = math.pi * (new_h ** 2) * (3 * new_r - new_h) / 3
    return new_Vseg


# сила Архимеда
def Arh_force(new_V):
    new_Farh = tp.water_p * tp.sea_g * new_V
    return new_Farh


def water_v(new_m, new_V, new_r):
    new_v = math.sqrt(
        ((new_m * tp.sea_g - tp.water_p * tp.sea_g * new_V) * 2) / (tp.Cf * tp.water_p * math.pi * new_r ** 2))
    return new_v


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
    global check, m_entry, V_entry, h_entry, h, m, V, V0, h_delta_entry, fly_arr_w, a0_arr_w, hit_arr_w, all_weight_arr, part_h_arr, part_water_ar_w, water_ar_w, all_h_arr, th, fly_arr_v, a0_arr_v, hit_arr_v, part_water_ar_v, water_ar_v, all_v_arr
    if not m_entry.is_focused and check[0] == 1:
        m_entry.set_text(limit_check(m_entry.get_text(), '1', '1', '1')[0])
        m = float(m_entry.get_text())
        check[0] = 0
    if not V_entry.is_focused and check[1] == 1:
        V_entry.set_text(limit_check('1', V_entry.get_text(), '1', '1')[1])
        V = float(V_entry.get_text())
        V0 = V
        check[1] = 0
    if not h_entry.is_focused and check[2] == 1:
        h_entry.set_text(limit_check('1', '1', h_entry.get_text(), '1')[2])
        h = float(h_entry.get_text())
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
def enable(yes):
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
    global out_f
    screen.blit(out_f.render('Ускорение    ' + str(round(a, 3)), False, tp.black), (50, tp.text_y))
    screen.blit(out_f.render('Вес  ' + str(round(weight, 3)), False, tp.black), (50, tp.text_y + 70))
    screen.blit(out_f.render('Скорось  ' + str(round(v1, 3)), False, tp.black), (50, tp.text_y + 70 * 2))
    screen.blit(out_f.render('Высота   ' + str(round(h, 3)), False, tp.black), (50, tp.text_y + 70 * 3))

    window.blit(out_f.render('м/с2', False, tp.black), (tp.text_x, tp.text_y))
    window.blit(out_f.render('Н', False, tp.black), (tp.text_x, tp.text_y + 70))
    window.blit(out_f.render('м/с', False, tp.black), (tp.text_x, tp.text_y + 70 * 2))
    window.blit(out_f.render('м', False, tp.black), (tp.text_x, tp.text_y + 70 * 3))


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
    if round(new_a, 1) == 0:  # or len(a0_arr_w) > 0:
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

    if len(part_water_ar_w) > 0:
        max_x = max([max(fly_arr_w), max(part_water_ar_w)]) * 1.2
    else:
        max_x = max(fly_arr_w)

    fig = plt.axis([0, max(all_h_arr) * 1.2 + 20, 0, max_x])
    plt.grid(True)
    plt.plot(part_h_arr[0], fly_arr_w, color='b', label='Полет с ускорением')

    if len(part_h_arr[1]) > 0:
        plt.plot([part_h_arr[0][-1], part_h_arr[1][0]], [fly_arr_w[-1], a0_arr_w[0]], color='g',
                 label='Полет без ускорения')
        plt.plot(part_h_arr[1], a0_arr_w, color='g')
        if len(part_h_arr[2]) > 0:
            plt.plot([part_h_arr[1][-1], part_h_arr[2][0]], [a0_arr_w[-1], hit_arr_w[0]], color='r')

    if len(part_h_arr[1]) == 0 and len(part_h_arr[2]) > 0:
        plt.plot([part_h_arr[0][-1], part_h_arr[2][0]], [fly_arr_w[-1], hit_arr_w[0]], color='r')

    if len(part_h_arr[2]) > 0:
        plt.plot(part_h_arr[2], hit_arr_w, color='r')
        if len(part_h_arr[3]) > 0:
            plt.text(part_h_arr[3][0], max(part_water_ar_w) * 1.2 - 5, f' Fудара = {round(hit_arr_w[-1])} Н',
                     bbox=tp.box)
            plt.plot([part_h_arr[2][-1], part_h_arr[3][0]], [hit_arr_w[-1], part_water_ar_w[0]], color='r',
                     label='удар о воду')

    if len(part_h_arr[3]) > 0:
        plt.plot(part_h_arr[3], part_water_ar_w, color='c', label='Неполное погружение')

    if len(part_h_arr[4]) > 0 and len(part_h_arr[3]) > 0:
        plt.plot([part_h_arr[3][-1], part_h_arr[4][0]], [part_water_ar_w[-1], water_ar_w[0]], color='m',
                 label='погружение')
        plt.plot(part_h_arr[4], water_ar_w, color='m')
    plt.legend()
    plt.xlabel("Высота(м)")  # подпишем оси
    plt.ylabel("Вес(Н)")

    h = float(h_entry.get_text())
    locs, labels = plt.xticks()
    plt.xticks(ticks=locs, labels=[round(h - i) for i in plt.xticks()[0]])

    plt.subplot(1, 2, 2)
    plt.grid(True)
    plt.plot(part_h_arr[0], fly_arr_v, color='b', label='Полет с ускорением')

    if len(part_h_arr[1]) > 0:
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


def Draw_animation(vertically):
    fig, ax = plt.subplots(figsize=(6.4 * 1.7, 4.8 * 1.7))
    ax = plt.axis([0, max(all_h_arr) * 1.1, 0, tp.max_m * 12])
    plt.xlabel('Путь')
    plt.ylabel('Вес')
    red_dot, = plt.plot([0], [0], 'ro')
    print(len(all_weight_arr), len(all_h_arr))
    plt.grid(True)

    def animate(i):
        red_dot.set_data(all_h_arr[i], all_weight_arr[i])
        line = plt.plot(all_h_arr[:i], all_weight_arr[:i], 'b')
        return red_dot,

    if len(all_h_arr) < 1500:
        ani = FuncAnimation(fig, animate, interval=10, repeat=True, frames=len(all_weight_arr))
        i = 1
        while True:
            if os.path.isfile(f'График {i}.png'):
                i += 1
            else:
                ani.save(f'Анимация {i}.gif', writer="pillow")
                break

    plt.close(fig)


def Reset_Variable():
    global weight, time, p, a, th, g, v0, v1, Vseg, Farh, P, R, all_weight_arr, part_h_arr, all_h_arr, fly_arr_w, a0_arr_w, hit_arr_w, part_water_ar_w, water_ar_w, fly_arr_v, a0_arr_v, hit_arr_v, part_water_ar_v, water_ar_v, all_v_arr
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
    fly_arr_w = [0]  # список веса для графика
    a0_arr_w = []
    hit_arr_w = []
    all_weight_arr = []
    part_water_ar_w = []
    water_ar_w = []
    fly_arr_v = [0]  # список скоростей для графика
    a0_arr_v = []
    hit_arr_v = []
    part_water_ar_v = []
    water_ar_v = []
    part_h_arr = [[0], [], [], [], []]
    all_weight_arr = [0]
    all_v_arr = [0]
    all_h_arr = [0]  # список высот для графика


def Correct_delta_h(new_h):
    if 0 < new_h <= 1:
        new_delta_h = new_h
    elif h <= 0:
        if 0 >= h > -10:
            new_delta_h = 1
        else:
            new_delta_h = 10
    else:
        if new_h // 2500 >= 1:
            new_delta_h = new_h // 2500
        else:
            new_delta_h = 1

    return new_delta_h


def Start_bf():
    global R, running_time
    if Start_b.text == 'Start':
        Start_b.set_text('Stop')
        R = round(((V * 3) / (math.pi * 4)) ** (1 / 3), 3)
        Error_msg.text = ''
        running_time = True
    else:
        Start_b.set_text('Start')
        running_time = False


def Restart_bf():
    global running_time, m, V, h, R, V0
    Start_b.set_text('Start')
    running_time = False
    Reset_Variable()
    m = float(m_entry.get_text())
    V = float(V_entry.get_text())
    V0 = V
    h = float(h_entry.get_text())
    R = round(((V * 3) / (math.pi * 4)) ** (1 / 3), 3)
    tp.delta_h = Correct_delta_h(h)  # функция определения шага от высоты
    pygame.draw.rect(screen, tp.white, (50, tp.text_y, 250, 230))
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
    Draw_graph(vertically, True, False)
    # Draw_animation()

    Start_b.enable()
    Restart_b.enable()
    m_entry.enable()
    V_entry.enable()
    h_entry.enable()

    h_delta_entry.disable()


def Graph_type_bf():
    global draw_weight_g
    if Graph_type_b.text == 'P(h)':
        Graph_type_b.set_text('v(h)')
        draw_weight_g = False
        return None
    if Graph_type_b.text == 'v(h)':
        Graph_type_b.set_text('P(h)')
        draw_weight_g = True
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


def Prev_point_bf():
    global running_time
    running_time = False


def Next_point_bf():
    global running_time
    running_time = False


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
    Draw_graph(vertically, False, True)
    if saving_anim:
        Draw_animation(vertically)
    Save_csv()


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
        h = float(h_entry.get_text())
        if vertically:
            max_x, max_y = correct_borders(max_x, max_y)

            for i in range(self.ab_len // tp.ab_interval + 1):
                text = h - max_x / self.ab_len * tp.ab_interval * i
                surface.blit(graph_f.render(f'{round(text)}', False, tp.black),
                             (self.ab_x + i * tp.ab_interval, self.ab_y + 5))

            for i in range(self.ord_len // tp.ord_interval + 1):
                text = max_y / self.ord_len * tp.ord_interval * i
                surface.blit(graph_f.render(f'{round(text)}', False, tp.black),
                             (self.ord_x - len(str(round(text))) * 12, self.ord_y - i * tp.ord_interval))
        else:
            max_y, max_x = correct_borders(max_x, max_y)

            for i in range(self.ab_len // tp.ab_interval + 1):
                text = max_x / self.ab_len * tp.ab_interval * i
                surface.blit(graph_f.render(f'{round(text)}', False, tp.black),
                             (self.ab_x + i * tp.ab_interval, self.ab_y + 5))

            for i in range(self.ord_len // tp.ord_interval + 1):
                text = h - (max_y - max_y / self.ord_len * tp.ord_interval * i)
                surface.blit(graph_f.render(f'{round(text)}', False, tp.black),
                             (self.ord_x - len(str(round(text))) * 12, self.ord_y - i * tp.ord_interval))


    def draw_weight_graph(self, surface, all_h, all_weight, max_x, max_y, vertically):
        if vertically:
            max_x, max_y = correct_borders(max_x, max_y)
            surface.blit(graph_f.render('высота', False, tp.black), (self.ab_x + self.ab_len + 10, self.ab_y - 25))
            surface.blit(graph_f.render('вес', False, tp.black), (self.ord_x - 1.5 * 10, self.ord_y - self.ord_len - 35))
            x = [i * (self.ab_len / max_x) for i in all_h]
            y = [i * (self.ord_len / max_y) for i in all_weight]
            for i in range(len(all_h) - 1):
                pygame.draw.line(surface, tp.red, (x[i] + self.ab_x, self.ab_y - y[i]),
                                 (x[i + 1] + self.ab_x, self.ab_y - y[i + 1]), width=3)
        else:
            max_y, max_x = correct_borders(max_x, max_y)
            surface.blit(graph_f.render('вес', False, tp.black), (self.ab_x + self.ab_len + 10, self.ab_y - 25))
            surface.blit(graph_f.render('высота', False, tp.black), (self.ord_x - 3 * 10, self.ord_y - self.ord_len - 35))
            x = [i * (self.ab_len / max_x) for i in all_weight]
            y = [self.ord_len - i * (self.ord_len / max_y) for i in all_h]
            for i in range(len(all_h) - 1):
                pygame.draw.line(surface, tp.red, (x[i] + self.ab_x, self.ab_y - y[i]),
                                 (x[i + 1] + self.ab_x, self.ab_y - y[i + 1]), width=3)


    def draw_v_graph(self, surface, all_h, all_v, max_x, max_y, vertically):
        if vertically:
            max_x, max_y = correct_borders(max_x, max_y)
            surface.blit(graph_f.render('высота', False, tp.black), (self.ab_x + self.ab_len + 10, self.ab_y - 25))
            surface.blit(graph_f.render('скорость', False, tp.black), (self.ord_x - 4 * 10, self.ord_y - self.ord_len - 35))
            x = [i * (self.ab_len / max_x) for i in all_h]
            y = [i * (self.ord_len / max_y) for i in all_v]
            for i in range(len(all_h) - 1):
                pygame.draw.line(surface, tp.blue, (x[i] + self.ab_x, self.ab_y - y[i]),
                                 (x[i + 1] + self.ab_x, self.ab_y - y[i + 1]), width=3)
        else:
            max_y, max_x = correct_borders(max_x, max_y)
            surface.blit(graph_f.render('скорость', False, tp.black), (self.ab_x + self.ab_len + 10, self.ab_y - 25))
            surface.blit(graph_f.render('высота', False, tp.black), (self.ord_x - 3 * 10, self.ord_y - self.ord_len - 35))
            x = [i * (self.ab_len / max_x) for i in all_v]
            y = [i * (self.ord_len / max_y) for i in all_h]
            for i in range(len(all_h) - 1):
                pygame.draw.line(surface, tp.blue, (x[i] + self.ab_x, self.ab_y - y[i]),
                                 (x[i + 1] + self.ab_x, self.ab_y - y[i + 1]), width=3)


'''Создаем окно, настраиваем шрифт и часы'''
pygame.init()
pygame.font.init()
out_f = pygame.font.Font(None, 30)
graph_f = pygame.font.Font(None, 25)
manager = pgui.UIManager((tp.WIDTH, tp.HEIGHT))
screen = pygame.display.set_mode((tp.WIDTH, tp.HEIGHT))
pygame.display.set_caption("Falling Ball")
all_sprites = pygame.sprite.Group()
clock = pygame.time.Clock()
screen.fill(tp.white)

'''Предварительно задаем переменные'''
m = 10
V = 10
V0 = 10
h = 200000
R = ((V * 3) / (math.pi * 4)) ** (1 / 3)
r = R
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
R = round(((V * 3) / (math.pi * 4)) ** (1 / 3), 3)

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

vertically = True
draw_weight_g = True
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
                                      text='Пред',
                                      manager=manager)
Next_point_b = pgui.elements.UIButton(relative_rect=pygame.Rect((665, 10), tp.Restart_b_size),
                                      text='След',
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
screen.blit(out_f.render(tp.title_m, False, tp.black), (50, 105))
screen.blit(out_f.render(tp.title_V, False, tp.black), (50, 175))
screen.blit(out_f.render(tp.title_h, False, tp.black), (50, 245))
screen.blit(out_f.render(tp.title_delta_h, False, tp.black), (50, 315))

screen.blit(graph_f.render(f'{tp.min_m}-{tp.max_m}', False, (30, 30, 30)), (tp.Input_x, tp.Input_y + 30))
screen.blit(graph_f.render(f'{tp.min_v}-{tp.max_v}', False, (30, 30, 30)), (tp.Input_x, tp.Input_y + 100))
screen.blit(graph_f.render(f'{tp.min_h}-{tp.max_h}', False, (30, 30, 30)), (tp.Input_x, tp.Input_y + 170))
screen.blit(out_f.render('~', False, (30, 30, 30)), (tp.Input_x + 5, tp.Input_y + 240))

run = True
while run:
    if len(all_h_arr) > 1:
        Draw_graph_b.enable()
        Next_point_b.enable()
        Prev_point_b.enable()
        Save_all_b.enable()
    else:
        Draw_graph_b.disable()
        Next_point_b.disable()
        Prev_point_b.disable()
        Save_all_b.disable()

    time_delta = clock.tick(60) / 1000.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

        if event.type == pygame.USEREVENT:
            if event.user_type == pgui.UI_BUTTON_PRESSED:  # обработка нажатий на кнопки
                if event.ui_element == Start_b:  # нажатие на старт
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
                    Next_point_bf()
                if event.ui_element == Prev_point_b:
                    Prev_point_bf()
                if event.ui_element == Graph_reverse_b:
                    Graph_reverse_bf()
                if event.ui_element == Save_all_b:
                    Save_all_bf()

        manager.process_events(event)
    if not running_time:
        enable(True)
        focus_check()
        changes()

    if running_time:
        enable(False)

        if h > 0:  # этап падения
            r = round(((V0 * 3) / (math.pi * 4)) ** (1 / 3), 3)
            S = round(math.pi * (r ** 2), 3)
            tp.delta_h = Correct_delta_h(h)
            v0 = v1
            g = g_h(h)
            p = p_h(h)
            weight = resist_force(v0, p, S, tp.Cf)
            a = a_F(g, weight)

            try:
                v1 = v_S(a, v0, tp.delta_h)
            except Exception:
                print(v1, v0, g, a, h, p, S, weight)
                Restart_bf()
                running_time = False
            arr_append(th, weight, a, tp.delta_h)
            all_weight_arr.append(weight)
            all_v_arr.append(v1)
            all_h_arr.append(th + tp.delta_h)

            if 200000 >= h > 1:
                h = round(h - tp.delta_h, 3)  # тякущая высота
                th = round(th + tp.delta_h, 3)  # сколько пролетел
            if 0 < h <= 1:
                th = round(float(h_entry.get_text()), 0)  # сколько пролетел
                h = 0  # тякущая высота

        if h == 0 and len(part_h_arr[2]) == 0:  # удар о воду
            tp.delta_h = 10
            v0 = v1
            weight = resist_force(v1, tp.water_p, S, tp.Cf)
            a = a_F(tp.sea_g, weight)
            v1 = 0
            time += tp.delta_t
            hit_arr_w.append(weight)
            hit_arr_v.append(v1)
            all_weight_arr.append(weight)
            all_v_arr.append(v1)
            all_h_arr.append(th)
            part_h_arr[2].append(th)

        if -(2 * R) <= h <= 0 and len(part_h_arr[2]) != 0:  # неполное погружение
            tp.delta_t = 0.005
            v0 = v1
            Vseg = round(Vseg_h(abs(h), R), 5)  # спросить про расчет объема частично погруженного шара
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
                running_time = False
                if saving_anim:
                    Draw_animation(vertically)
                Save_csv()
                Draw_graph(vertically, True, True)

        if -tp.depth <= h < -(2 * R):  # погружение
            tp.delta_h = Correct_delta_h(h)

            if tp.delta_h > tp.depth + h:
                tp.delta_h = tp.depth + h

            v0 = v1
            P = P_h(abs(h))
            if len(water_ar_w) > 0:
                V = comress_h(abs(h), V0, r)  # ???
            else:
                V = Vseg
            print('1', h, V)
            r = round(((V * 3) / (math.pi * 4)) ** (1 / 3), 3)
            S = round(math.pi * (r ** 2), 3)
            Farh = Arh_force(V)
            weight = round(m * tp.sea_g - Farh, 4)
            print('2', h, V, V0, m * tp.sea_g, Farh, v1)

            v1 = water_v(m, V, r)
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
            running_time = False
            if saving_anim:
                Draw_animation(vertically)
            Save_csv()
            Draw_graph(vertically, True, True)

        pygame.draw.rect(screen, tp.white, (50, tp.text_y, 250, 230))
        draw_text(screen)

    surface1.fill((240, 240, 240))
    axis1.draw(surface1)
    if len(all_h_arr) > 1:
        if draw_weight_g:
            axis1.sign_devision(surface1, max(all_h_arr), max(all_weight_arr), vertically)
            axis1.draw_weight_graph(surface1, all_h_arr, all_weight_arr, max(all_h_arr), max(all_weight_arr),
                                    vertically)
        else:
            axis1.sign_devision(surface1, max(all_h_arr), max(all_v_arr), vertically)
            axis1.draw_v_graph(surface1, all_h_arr, all_v_arr, max(all_h_arr), max(all_v_arr), vertically)

    manager.update(time_delta)
    manager.draw_ui(screen)
    all_sprites.draw(screen)

    screen.blit(surface1, (350, 100))
    clock.tick(tp.FPS)
    pygame.display.flip()
