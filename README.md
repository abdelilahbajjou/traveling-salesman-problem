# Traveling Salesman Problem (TSP) Solver

A comprehensive Python implementation of the Traveling Salesman Problem with multiple algorithms and interactive visualization.

## Features

- **Multiple Algorithms**: 
  - Brute Force (Exact solution)
  - Branch and Bound (Exact solution with pruning)
  - Nearest Neighbor (Heuristic approximation)

- **Interactive Visualization**: 
  - Real-time animation of algorithm execution
  - African map background with city locations
  - Adjustable animation speed
  - Distance matrix visualization

- **GUI Interface**: 
  - Easy-to-use Tkinter interface
  - Algorithm selection dropdown
  - Real-time results display

## Screenshots

![TSP Visualization](screenshots/tsp_visualization.png)
*TSP algorithm visualization on African map*

## Installation

### Prerequisites
- Python 3.7+
- Required packages (install via pip):

```bash
pip install -r requirements.txt
```

### Dependencies
- `tkinter` (usually comes with Python)
- `matplotlib`
- `numpy`
- `networkx`

## Usage

### Running the Application

```bash
python tsp_solver.py
```

### Using the GUI

1. **Select Algorithm**: Choose from Brute Force, Branch and Bound, or Nearest Neighbor
2. **Click "Solve TSP"**: The visualization window will open
3. **Adjust Speed**: Use the slider to control animation speed
4. **Start Animation**: Click the "Start" button to begin the visualization

### Algorithm Comparison

| Algorithm | Time Complexity | Space Complexity | Optimal Solution |
|-----------|----------------|------------------|------------------|
| Brute Force | O(n!) | O(n) | ✅ Yes |
| Branch and Bound | O(n!) worst case | O(n) | ✅ Yes |
| Nearest Neighbor | O(n²) | O(n) | ❌ Approximate |

## Project Structure

```
tsp-solver/
├── tsp_solver.py          # Main application file
├── map_image.png          # African map background
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── screenshots/          # Application screenshots
└── docs/                 # Documentation (French report)
    └── rapport_tsp.pdf   # Project report in French
```

## Data Structure

The project uses a 6-city example with African locations:
- Node 0: Northwest Africa (Algeria/Morocco region)
- Node 1: Central Africa (Niger/Chad region)
- Node 2: West Africa (Mali/Nigeria region)
- Node 3: North-Central Africa (Libya region)
- Node 4: East Africa (Egypt/Sudan region)
- Node 5: Southwest Africa (Ghana region)

## Algorithms Explained

### Brute Force
- Evaluates all possible permutations
- Guarantees optimal solution
- Best for small datasets (< 10 cities)

### Branch and Bound
- Uses pruning to eliminate suboptimal branches
- Guarantees optimal solution
- More efficient than brute force for medium datasets

### Nearest Neighbor
- Greedy algorithm: always go to nearest unvisited city
- Fast execution
- Provides good approximation but not always optimal

## Visualization Features

- **Real-time Animation**: Watch algorithms explore different routes
- **Interactive Controls**: Adjust speed and start/stop animation
- **Distance Display**: See current tour distance and best route
- **Color Coding**: 
  - Red: Start city
  - Green: End city
  - Blue: Current path
  - Dark Blue: Best path found

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Future Enhancements

- [ ] Add more heuristic algorithms (Genetic Algorithm, Simulated Annealing)
- [ ] Support for custom distance matrices
- [ ] Import/export functionality for city data
- [ ] Performance benchmarking tools
- [ ] 3D visualization option

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- African map data for visualization
- NetworkX library for graph operations
- Matplotlib for visualization capabilities

## Contact

For questions or suggestions, please open an issue on GitHub.

---

*This project was developed as part of an academic assignment on optimization algorithms.*