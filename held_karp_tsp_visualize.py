import argparse
import math
import random
from itertools import combinations

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def reconstruct_partial_tour(mask, last, parent):
    tour = [last]
    while tour[-1] != 0:
        prev = parent.get((mask, tour[-1]))
        if prev is None:
            break
        mask ^= 1 << tour[-1]
        tour.append(prev)
    return list(reversed(tour))


def held_karp(points, record_steps=False):
    n = len(points)
    if n <= 1:
        if record_steps:
            return 0.0, [0], []
        return 0.0, [0]

    # Precompute distances
    dist = [[distance(points[i], points[j]) for j in range(n)] for i in range(n)]

    # DP table: dp[mask][j] = min cost to start at 0, visit set mask, and end at j
    dp = {}
    parent = {}
    steps = []

    # Base case: only the start node visited and current node is start (0)
    dp[(1 << 0, 0)] = 0.0

    for subset_size in range(2, n + 1):
        for subset in combinations(range(1, n), subset_size - 1):
            mask = 1 << 0
            for bit in subset:
                mask |= 1 << bit

            for j in subset:
                prev_mask = mask ^ (1 << j)
                best_cost = float('inf')
                best_prev = None
                for k in range(n):
                    if k == j or not (prev_mask & (1 << k)):
                        continue
                    cost = dp.get((prev_mask, k), float('inf')) + dist[k][j]
                    if cost < best_cost:
                        best_cost = cost
                        best_prev = k
                dp[(mask, j)] = best_cost
                parent[(mask, j)] = best_prev
                if record_steps:
                    steps.append({
                        'subset_size': subset_size,
                        'mask': mask,
                        'end': j,
                        'path': reconstruct_partial_tour(mask, j, parent),
                        'cost': best_cost,
                    })

    full_mask = (1 << n) - 1
    best_cost = float('inf')
    best_end = None
    best_full_steps = []
    for j in range(1, n):
        tour_cost = dp[(full_mask, j)] + dist[j][0]
        if tour_cost < best_cost:
            best_cost = tour_cost
            best_end = j
        if record_steps:
            best_full_steps.append({
                'subset_size': n,
                'mask': full_mask,
                'end': j,
                'path': reconstruct_partial_tour(full_mask, j, parent),
                'cost': tour_cost,
                'is_complete': True,
                'is_best_full': tour_cost == best_cost,
            })

    if record_steps:
        steps.extend(best_full_steps)

    tour = [0]
    mask = full_mask
    last = best_end
    for _ in range(n - 1):
        tour.append(last)
        prev = parent[(mask, last)]
        mask ^= 1 << last
        last = prev
    tour.append(0)
    tour.reverse()

    if record_steps:
        return best_cost, tour, steps
    return best_cost, tour


def animate_held_karp(points, steps, title=None, speed=1.0):
    xs, ys = zip(*points)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(xs, ys, c='tab:blue', s=80, zorder=3)

    for i, (x, y) in enumerate(points):
        ax.text(x, y, f' {i}', fontsize=10, zorder=4)

    current_line, = ax.plot([], [], '-o', c='tab:green', linewidth=2, markersize=6, alpha=0.7)
    best_line, = ax.plot([], [], '-o', c='tab:orange', linewidth=2, markersize=6, alpha=0.9)
    highlight_line, = ax.plot([], [], '-', c='tab:red', linewidth=3, alpha=0.6)

    status_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top')
    title_text = ax.set_title(title or 'Held-Karp TSP Search Animation')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.4)

    axpause = plt.axes([0.75, 0.02, 0.1, 0.05])
    axspeed = plt.axes([0.1, 0.02, 0.55, 0.05])
    pause_button = Button(axpause, 'Pause')
    speed_slider = Slider(axspeed, 'Speed', 0.25, 3.0, valinit=speed, valstep=0.25)

    state = {'running': True, 'interval': max(50, int(400 / speed)), 'best_tour': None, 'best_cost': float('inf')}

    def draw_path(line_obj, path, closed=False):
        if not path:
            line_obj.set_data([], [])
            return
        x = [points[i][0] for i in path]
        y = [points[i][1] for i in path]
        if closed and len(path) > 1:
            x.append(points[path[0]][0])
            y.append(points[path[0]][1])
        line_obj.set_data(x, y)

    def update(frame):
        step = steps[frame]
        draw_path(current_line, step['path'])

        status = f"Subset size: {step['subset_size']} | end={step['end']} | cost={step['cost']:.4f}"
        if step.get('is_complete'):
            status += ' | candidate full tour'
            draw_path(highlight_line, step['path'] + [0], closed=False)
            if step.get('is_best_full'):
                state['best_tour'] = step['path'] + [0]
                state['best_cost'] = step['cost']
                draw_path(best_line, state['best_tour'], closed=False)
        else:
            highlight_line.set_data([], [])

        status_text.set_text(status)
        return current_line, best_line, highlight_line, status_text

    def toggle(event):
        if state['running']:
            anim.event_source.stop()
            pause_button.label.set_text('Play')
        else:
            anim.event_source.start()
            pause_button.label.set_text('Pause')
        state['running'] = not state['running']

    def change_speed(val):
        state['interval'] = max(50, int(400 / val))
        anim.event_source.interval = state['interval']

    pause_button.on_clicked(toggle)
    speed_slider.on_changed(change_speed)

    anim = FuncAnimation(fig, update, frames=len(steps), interval=state['interval'], repeat=False)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show()


def visualize_tour(points, tour, title=None):
    xs, ys = zip(*points)
    plt.figure(figsize=(8, 8))
    plt.scatter(xs, ys, c='tab:blue', s=80, zorder=3)

    for i, (x, y) in enumerate(points):
        plt.text(x, y, f' {i}', fontsize=10, zorder=4)

    path_x = [points[i][0] for i in tour]
    path_y = [points[i][1] for i in tour]
    plt.plot(path_x, path_y, '-o', c='tab:orange', linewidth=2, markersize=6)

    plt.title(title or 'Held-Karp TSP Tour')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.show()


def sample_points(n, seed=None):
    rng = random.Random(seed)
    return [(rng.random() * 10, rng.random() * 10) for _ in range(n)]


def parse_args():
    parser = argparse.ArgumentParser(description='Held-Karp TSP solver and visualizer')
    parser.add_argument('-n', '--num-cities', type=int, default=10, help='Number of cities to generate')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducible cities')
    parser.add_argument('--coords', nargs='+', type=float, help='Direct list of x y coordinates for cities')
    parser.add_argument('--animate', action='store_true', help='Play an animation of the Held-Karp search')
    parser.add_argument('--speed', type=float, default=1.0, help='Animation speed multiplier (higher is faster)')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.coords:
        if len(args.coords) % 2 != 0:
            raise ValueError('Coordinates must be provided as x y pairs.')
        points = [(args.coords[i], args.coords[i + 1]) for i in range(0, len(args.coords), 2)]
    else:
        points = sample_points(args.num_cities, seed=args.seed)

    if args.animate:
        cost, tour, steps = held_karp(points, record_steps=True)
    else:
        cost, tour = held_karp(points)

    print(f'Optimal tour cost: {cost:.4f}')
    print('Tour order:', ' -> '.join(map(str, tour)))

    if args.animate:
        animate_held_karp(points, steps, title=f'Held-Karp TSP Search (cost={cost:.4f})', speed=args.speed)
    else:
        visualize_tour(points, tour, title=f'Held-Karp TSP (cost={cost:.4f})')


if __name__ == '__main__':
    main()
