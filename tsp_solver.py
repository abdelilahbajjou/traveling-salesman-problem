import tkinter as tk
import itertools
import heapq
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from tkinter import ttk, Frame
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Slider, Button




# Distance Matrix (Static Example)
distance_matrix = np.array([
    [0, 10, 15, 20, 6, 10],
    [10, 0, 35, 25, 12, 15],
    [15, 35, 0, 30, 8, 12],
    [10, 16, 30, 15, 2, 1],
    [6, 12, 8, 10, 0, 7],
    [10, 15, 12, 13, 7, 0]
])

city_coords = [
    (10.0, -5.0),  # Node 0 (Northwest Africa, near Algeria/Morocco region)
    (1.0, 20.0),   # Node 1 (Central Africa, near Niger/Chad region)
    (-5.0, 20.0),   # Node 2 (West Africa, near Mali/Nigeria region)
    (15.0, 10.0),   # Node 3 (North-Central Africa, near Libya region)
    (10.0, 35.0),   # Node 4 (East Africa, near Egypt/Sudan region)
    (-2.0, 3.0)     # Node 5 (Southwest Africa, near Ghana region)
]








# Normalize city coordinates for visualization
def normalize_coordinates(coords, xlim=(-1.4, 1.4), ylim=(-1.4, 1.4)):
    lats, lons = zip(*coords)
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    normalized = {}
    for i, (lat, lon) in enumerate(coords):
        # Adjust normalization to provide better spacing for nodes
        x = xlim[0] + (lon - min_lon) / (max_lon - min_lon) * (xlim[1] - xlim[0])
        y = ylim[0] + (lat - min_lat) / (max_lat - min_lat) * (ylim[1] - ylim[0])
        normalized[i] = (x, y)
    return normalized

# Generate positions for nodes using normalized city coordinates
pos = normalize_coordinates(city_coords)


# Brute Force Algorithm
def brute_force_tsp(matrix):
    n = len(matrix)
    all_permutations = itertools.permutations(range(1, n))  # Start from 1 to avoid duplicating start node
    min_distance = float('inf')
    best_tour = None
    distances = []

    for perm in all_permutations:
        # Include the start (0) and end (0) nodes in the tour
        full_tour = (0,) + perm + (0,)
        tour_distance = sum(matrix[full_tour[i], full_tour[i + 1]] for i in range(len(full_tour) - 1))
        distances.append((tour_distance, full_tour))
        if tour_distance < min_distance:
            min_distance = tour_distance
            best_tour = full_tour

    return best_tour, min_distance, distances

# Branch and Bound Algorithm
def branch_and_bound(matrix):
    n = len(matrix)
    pq = []
    heapq.heappush(pq, (0, [0]))  # Start at city 0
    best_cost = float('inf')
    best_path = None
    distances = []

    while pq:
        current_cost, current_path = heapq.heappop(pq)
        if len(current_path) == n:  # Completed a tour
            total_cost = current_cost + matrix[current_path[-1]][current_path[0]]
            distances.append((total_cost, current_path + [0]))
            if total_cost < best_cost:
                best_cost = total_cost
                best_path = current_path + [0]
            continue

        for next_city in range(n):
            if next_city not in current_path:
                new_cost = current_cost + matrix[current_path[-1], next_city]
                if new_cost < best_cost:
                    heapq.heappush(pq, (new_cost, current_path + [next_city]))
                    distances.append((new_cost, current_path + [next_city]))

    return best_path, best_cost, distances

# Nearest Neighbor Algorithm
def nearest_neighbor_algorithm(matrix):
    n = len(matrix)
    current_city = 0
    tour = [current_city]
    total_distance = 0
    distances = []
    unvisited = set(range(n)) - {current_city}

    while unvisited:
        nearest_city = min(unvisited, key=lambda city: matrix[current_city][city])
        total_distance += matrix[current_city][nearest_city]
        current_city = nearest_city
        tour.append(current_city)
        unvisited.remove(current_city)
        distances.append((total_distance, tour[:]))

    total_distance += matrix[current_city][tour[0]]
    tour.append(tour[0])
    distances.append((total_distance, tour[:]))
    return tour, total_distance, distances


def visualize_tsp_animation(best_tour, distances, distance_matrix):
    fig, ax = plt.subplots(figsize=(10, 6))
    map_image = plt.imread("map_image.png")
    ax.imshow(map_image, extent=[-1.5, 1.5, -1.5, 1.5], aspect='auto')

    # Create a graph with nodes and positions
    G = nx.Graph()
    n = len(distance_matrix)
    G.add_nodes_from(range(len(city_coords)))
    pos = normalize_coordinates(city_coords)  # Generate positions for nodes
    best_tour_reached = False  # Initialize the variable outside the update function

    # Add the speed slider directly in the visualization window
    slider_ax = plt.axes([0.2, 0.01, 0.6, 0.03])  # Position slider (left, bottom, width, height)
    speed_slider = Slider(slider_ax, 'Speed (ms)', 200, 5000, valinit=1000)

    ax.set_title("Traveling Salesman Problem Visualization in Africa", fontsize=16, color='darkblue')
    # Add a Start button
    start_button_ax = plt.axes([0.85, 0.01, 0.1, 0.04])  # Button position
    start_button = Button(start_button_ax, "Start")

    ani = None  # Placeholder for the animation

    # Draw all static elements (background edges and distances)
    for i in range(n):
        for j in range(i + 1, n):
            # Draw static edges
            ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color='black', linestyle='dashed', alpha=0.5)
            # Add distance labels
            edge_mid = [(pos[i][0] + pos[j][0]) / 2, (pos[i][1] + pos[j][1]) / 2]
            ax.text(edge_mid[0], edge_mid[1], f"{distance_matrix[i, j]:.2f}", fontsize=8, ha='center', color='darkred')

    for node in G.nodes:
        x, y = pos[node]
        ax.scatter(x, y, s=400, color='lightblue', edgecolors='black', linewidth=1.5, zorder=5)  # Smaller nodes
        ax.text(x, y, f"{node}", fontsize=8, ha='center', va='center', color='black', zorder=6)  # Node numbers


    def update(frame):
        nonlocal best_tour_reached  # Use the variable from the enclosing scope
        ax.clear()

        # Redraw the background map and static edges
        ax.imshow(map_image, extent=[-1.5, 1.5, -1.5, 1.5], aspect='auto')
        for i in range(n):
            for j in range(i + 1, n):
                ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color='black', linestyle='dashed', alpha=0.5)
                edge_mid = [(pos[i][0] + pos[j][0]) / 2, (pos[i][1] + pos[j][1]) / 2]
                ax.text(edge_mid[0], edge_mid[1], f"{distance_matrix[i, j]:.2f}", fontsize=12, ha='center', color='black')

        current_distance, current_tour = distances[frame]
        num_cities = len(current_tour)

        # Highlight the current path
        edges = [(current_tour[i], current_tour[i + 1]) for i in range(num_cities - 1)]
        for edge in edges:
            start, end = edge
            ax.plot([pos[start][0], pos[end][0]], [pos[start][1], pos[end][1]], color='blue', linewidth=3.5, alpha=0.8)

        ax.set_title("Traveling Salesman Problem Visualization in Africa", fontsize=16, color='darkblue')

        # Draw nodes with improved styling
        for node in G.nodes:
            x, y = pos[node]
            if node == current_tour[0]:  # Start node
                ax.scatter(x, y, s=400, color='red', edgecolors='black', linewidth=1.5, zorder=5)  # Larger red circle
                ax.text(x, y, f"{node}", fontsize=10, ha='center', va='center', color='black', zorder=6)  # Number inside node
            elif node == current_tour[-1]:  # End node
                ax.scatter(x, y, s=400, color='green', edgecolors='black', linewidth=1.5, zorder=5)  # Larger green circle
                ax.text(x, y, f"{node}", fontsize=10, ha='center', va='center', color='black', zorder=6)  # Number inside node
            else:  # Intermediate nodes
                ax.scatter(x, y, s=350, color='lightblue', edgecolors='black', linewidth=2, zorder=5)  # Intermediate blue circle
                ax.text(x, y, f"{node}", fontsize=10, ha='center', va='center', color='black', zorder=6)  # Number inside node

        # Highlight the best tour
        if len(current_tour) == len(best_tour) and current_tour == best_tour and not best_tour_reached:
            best_tour_reached = True
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='darkblue', width=3.5, alpha=1.0, ax=ax)

            # Add arrows for the best route
            for i in range(len(current_tour) - 1):
                start, end = current_tour[i], current_tour[i + 1]
                edge_mid = (
                    (pos[start][0] + pos[end][0]) / 2,
                    (pos[start][1] + pos[end][1]) / 2
                )
                arrow_pos = (
                    (edge_mid[0] + pos[end][0]) / 2,
                    (edge_mid[1] + pos[end][1]) / 2
                )
                ax.annotate('',
                            xy=arrow_pos, xytext=edge_mid,
                            arrowprops=dict(arrowstyle='->', color='darkblue', lw=2.0))

            # Display the best route
            best_route_str = " -> ".join(map(str, best_tour))
            fig.text(0.5, 0.06, f"Best route: {best_route_str} | Tour Distance: {current_distance:.2f}",
            fontsize=12, ha='center', color='black', bbox=dict(facecolor='white', alpha=0.8))

            if current_tour == best_tour:
                ani.event_source.stop()

        # Display current distance
        ax.text(1.1, -1.5, f"Tour Distance: {current_distance}", fontsize=14, ha='center', color='black')
    # Set up the animation
     # Start Animation Function
    def start_animation(event):
        nonlocal ani
        if ani is not None:
            ani.event_source.stop()  # Stop previous animation if any
        interval = speed_slider.val  # Get speed from slider
        ani = FuncAnimation(fig, update, frames=len(distances), interval=interval, repeat=False)
        plt.draw()

    start_button.on_clicked(start_animation)  # Connect the Start button

    plt.show()







# Function to Run Selected Algorithm
def run_algorithm(matrix, algorithm):
    if algorithm == "Brute Force":
        return brute_force_tsp(matrix)
    elif algorithm == "Branch and Bound":
        return branch_and_bound(matrix)
    elif algorithm == "Nearest Neighbor":
        return nearest_neighbor_algorithm(matrix)
    else:
        return None, None, None

##################################################################


#########################################################
# Tkinter Setup
def solve_tsp():
    algorithm = algo_var.get()
    best_tour, min_distance, distances = run_algorithm(distance_matrix, algorithm)
    if best_tour:

        visualize_tsp_animation(best_tour, distances, distance_matrix)
        print(f"Best tour: {best_tour}")
        print(f"Minimum distance: {min_distance}")
    else:
        result_label.config(text="Algorithm not implemented yet.")


# Main Window
window = tk.Tk()
window.title("Traveling Salesman Problem Solver")
window.geometry("900x600")  # Wider window for a better layout

# Apply a modern style
style = ttk.Style()
style.configure("TFrame", background="#f5f5f5")
style.configure("TButton", font=("Helvetica", 12), background="#4CAF50", foreground="white")
style.configure("TLabel", background="#f5f5f5", font=("Helvetica", 12))
style.configure("TCombobox", font=("Helvetica", 12))

# Main Frame
frame = ttk.Frame(window, padding="20")
frame.pack(pady=20)

# Dropdown Label
label = ttk.Label(frame, text="Select Algorithm:")
label.grid(row=0, column=0, padx=10, pady=5)

# Combobox for Algorithm Selection
algo_var = tk.StringVar()
algo_dropdown = ttk.Combobox(frame, textvariable=algo_var, state="readonly", font=("Helvetica", 12), width=25)
algo_dropdown['values'] = ["Brute Force", "Branch and Bound", "Nearest Neighbor"]
algo_dropdown.current(0)  # Default selection
algo_dropdown.grid(row=0, column=1, padx=10, pady=5)

# Solve Button
solve_button = tk.Button(window, text="Solve TSP", command=solve_tsp)
solve_button.pack(pady=20)

# Result Label
result_label = ttk.Label(window, text="", font=("Helvetica", 14), foreground="#333333")
result_label.pack(pady=10)

speed_var = tk.IntVar(value=500)  # Default speed (milliseconds)


window.mainloop()