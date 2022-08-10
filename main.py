import numpy as np
import math
import pygame
from utils import line_intersection, hls_to_rgb
import random
from dna import DNA
from neural_network import forward, eitan, relu


def reverse_interpolation(x, y):
    return (x @ y) / (y @ y)


def distance_point_to_segment(point, edge1, edge2):
    point = np.array(point) - edge1
    edge = np.array(edge2) - edge1
    t = np.clip(reverse_interpolation(point, edge), 0, 1)
    return np.linalg.norm(point - t * edge)


time = 0
WIDTH = 800
HEIGHT = 600

pygame.init()

screen = pygame.display.set_mode([WIDTH, HEIGHT])
infinity = float('inf')




screen_edges = [
    # screen edges
    ((0, 1), (WIDTH - 1, 1)),
    ((0, 1), (0, HEIGHT - 1)),
    ((0, HEIGHT - 1), (WIDTH - 1, HEIGHT - 1)),
    ((WIDTH - 1, 0), (WIDTH - 1, HEIGHT - 1)),
]

circle_map = [
    *screen_edges,
    # two big "third" lines
    ((WIDTH / 3, 0), (WIDTH / 3, HEIGHT * 2 / 3)),
    ((WIDTH * 2 / 3, HEIGHT), (WIDTH * 2 / 3, HEIGHT / 3)),
    # left broken lines:
    ((WIDTH / 6, HEIGHT / 5), (WIDTH / 6, HEIGHT * 2 / 5)),
    ((WIDTH / 6, HEIGHT * 3 / 5), (WIDTH / 6, HEIGHT * 4 / 5)),
    # middle broken lines:
    ((WIDTH / 2, HEIGHT / 5), (WIDTH / 2, HEIGHT * 2 / 5)),
    ((WIDTH / 2, HEIGHT * 3 / 5), (WIDTH / 2, HEIGHT * 4 / 5)),
    # right broken lines:
    ((WIDTH * 5 / 6, HEIGHT / 5), (WIDTH * 5 / 6, HEIGHT * 2 / 5)),
    ((WIDTH * 5 / 6, HEIGHT * 3 / 5), (WIDTH * 5 / 6, HEIGHT * 4 / 5)),
    # those must stay last
    # small diagonal thingies:
    ((WIDTH * 7.5 / 12, HEIGHT * 2 / 10), (WIDTH * 8.5 / 12, HEIGHT * 1 / 10)),
    ((WIDTH * 3.5 / 12, HEIGHT * 8 / 10), (WIDTH * 4.5 / 12, HEIGHT * 9 / 10)),
]

maze_map = [
    *screen_edges,

    ((.1 * WIDTH, .14 * HEIGHT), (.1 * WIDTH, .85 * HEIGHT)),
    ((.2 * WIDTH, HEIGHT), (.2 * WIDTH, .29 * HEIGHT)),
    ((.3 * WIDTH, .14 * HEIGHT), (.3 * WIDTH, .85 * HEIGHT)),
    ((.4 * WIDTH, HEIGHT), (.4 * WIDTH, .29 * HEIGHT)),
    ((.5 * WIDTH, .14 * HEIGHT), (.5 * WIDTH, .85 * HEIGHT)),
    ((.6 * WIDTH, HEIGHT), (.6 * WIDTH, .29 * HEIGHT)),
    ((.7 * WIDTH, .14 * HEIGHT), (.7 * WIDTH, .85 * HEIGHT)),
    ((.8 * WIDTH, HEIGHT), (.8 * WIDTH, .29 * HEIGHT)),
    ((.9 * WIDTH, .14 * HEIGHT), (.9 * WIDTH, .85 * HEIGHT)),

    ((.1 * WIDTH, .14 * HEIGHT), (.9 * WIDTH, .14 * HEIGHT))
]

death_lines = maze_map


pygame.font.init()
my_font = pygame.font.SysFont('Comic Sans MS', 30)


def resize_segment(segment):
    start, end = map(np.array, segment)
    facti_factor = (time % 2) / (random.random() * 4 + 6)
    return start + facti_factor, end - facti_factor

def rotate_segment(segment, angle=0.01):
    start, end = map(np.array, segment)
    mid = (start + end) / 2
    end -= mid
    start -= mid
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    start = rotation_matrix @ start
    end = rotation_matrix @ end
    return start + mid, end + mid

def move_them_thingies():
    global death_lines
    num_to_rotate = 8
    angle  = 0.01 if time % 200 < 100 else -0.01
    for i in range(len(death_lines) - num_to_rotate, len(death_lines)):
        death_lines[i] = rotate_segment(death_lines[i], angle)
        # death_lines[i] = resize_segment(death_lines[i])



class Amoeba:

    def __init__(self, dna=None) -> None:
        self.RADIUS = 25
        self.SPEED = 5
        self.angle = np.random.random() * 2 * math.pi
        self.died_at = infinity  # easier to compare to

        if dna is None:
            dna = DNA.new_random()

        self.dna = dna
        self.sight_angel = dna.sight_angle
        self.color = hls_to_rgb(dna.hue % 360, .7, .7)

        while True:
            loc_x = random.randint(self.RADIUS + 1, WIDTH - self.RADIUS - 1)
            loc_y = random.randint(self.RADIUS + 1, HEIGHT - self.RADIUS - 1)
            self.location = np.array([loc_x, loc_y])
            if all(distance_point_to_segment(self.location, *seg) > self.RADIUS for seg in death_lines):
                break
            


    def _get_sensor_edges_helper(self, _angle):
        end_x = math.cos(_angle) * (self.dna.sens_len + self.RADIUS)
        end_y = math.sin(_angle) * (self.dna.sens_len + self.RADIUS)

        start_x = math.cos(_angle) * self.RADIUS
        start_y = math.sin(_angle) * self.RADIUS

        return np.array([start_x, start_y]) + self.location, np.array([end_x, end_y]) + self.location


    def _read_sensor(self, sensor):
        min_dist = infinity
        for edge in death_lines:
            int_pnt = line_intersection(sensor, edge)
            if not int_pnt: continue
            sensor_start = sensor[0]
            dist = np.linalg.norm(sensor_start - int_pnt)
            min_dist = min(min_dist, dist)
    
        return 0 if math.isinf(min_dist) else 1 - min_dist / self.dna.sens_len

    def get_sensors_edges(self):
        cells_diff = self.sight_angel
        eye_1_1 = self.angle + (self.dna.sens_diff + cells_diff) / 2
        eye_1_2 = self.angle + (self.dna.sens_diff - cells_diff) / 2
        eye_2_1 = self.angle - (self.dna.sens_diff + cells_diff) / 2
        eye_2_2 = self.angle - (self.dna.sens_diff - cells_diff) / 2
        return map(self._get_sensor_edges_helper, (eye_1_1, eye_1_2, eye_2_1, eye_2_2))
        

    def sight(self) -> list[int]:
        return np.array(list(map(self._read_sensor, self.get_sensors_edges())))


    def think_about_sensors_meaning(self):
        # todo always use eitan
        return forward(self.sight(), *self.dna.weights_and_biases)


    def draw_me(self, draw_sensors=False):
        pygame.draw.circle(screen, self.color, self.location, self.RADIUS)

        eye_1_1, eye_1_2, eye_2_1, eye_2_2 = self.get_sensors_edges()

        # draw eyes!
        for sensor in (eye_1_1, eye_2_2):
            start, end = sensor
            drct = (end - start)
            drct /= np.linalg.norm(drct)
            where = drct * self.RADIUS + self.location
            pygame.draw.circle(screen, (255, 255, 255), where - drct * self.RADIUS * .25, self.RADIUS * .3)
            pygame.draw.circle(screen, (0, 0, 0), where - drct * self.RADIUS * .25, self.RADIUS * .15)

        if draw_sensors:
            color = (70, 70, 0)
            for sensor in (eye_1_1, eye_1_2, eye_2_1, eye_2_2):
                pygame.draw.line(screen, color, *sensor)


    def does_touch_death_lines(self):
        for line in death_lines:
            if distance_point_to_segment(self.location, *line) < self.RADIUS:
                return True
        return False

    
    def move_and_maybe_die(self):
        assert self.died_at == infinity

        delta_angle = self.think_about_sensors_meaning() * self.SPEED / 12 # the faster they are the bigger their reaction
        self.angle += delta_angle

        dx = math.cos(self.angle) * self.SPEED
        dy = math.sin(self.angle) * self.SPEED
        self.location[0] += dx
        self.location[1] += dy
        
        if self.does_touch_death_lines():
            self.died_at = time




def generate_population(amount, dna_to_mutate=None):
    if type(dna_to_mutate) is str:
        dna_to_mutate = DNA(dna_to_mutate)
    return tuple(
        Amoeba(dna=dna_to_mutate.mutation() if dna_to_mutate is not None else DNA.new_random())
        for _ in range(amount)
    )



def do_evolution(num_goods=5, num_bads=1, num_clones_each=5):
    """ returns a new set of amoebas based on the fittest from the current set """

    ams = sorted(amoebas, key=lambda am: am.died_at)
    results = []

    def clone_add_and_stuff(am):
        am.died_at = infinity
        results.append(am)
        for _ in range(num_clones_each):
            clone = Amoeba(dna=am.dna.mutation())
            results.append(clone)
    
    for _ in range(num_goods):
        am = ams.pop()
        clone_add_and_stuff(am)
       
    for _ in range(num_bads):
        am = random.choice(ams)
        ams.remove(am)
        clone_add_and_stuff(am)

    return tuple(results)




########################################
with open('./successful_dna.txt') as f:
    inital_dna = f.read().strip()

amoebas = generate_population(10, inital_dna)



def run(fps):
    global time, amoebas
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    print(amoebas[0].dna, '\n')


        screen.fill((0, 0, 0))

        for line in death_lines:
            pygame.draw.line(screen, (255, 255, 255), *line)

        num_alive = 0
        for amoeba in amoebas:
            if amoeba.died_at == infinity:
                num_alive += 1
                amoeba.move_and_maybe_die()
                amoeba.draw_me(draw_sensors=True)

        if num_alive <= 4:
            amoebas = do_evolution(num_goods=3, num_bads=0, num_clones_each=4)
        
        text = f"TIME: {time}"
        text_surface = my_font.render(text, False, (40, 120, 180))
        screen.blit(text_surface, (20, HEIGHT - 60))

        pygame.display.flip()
        clock.tick(fps)
        time += 1

        #move_them_thingies()


run(fps=40)
pygame.quit()
