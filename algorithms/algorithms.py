import heapq
from collections import deque, defaultdict
import math

class CityGraph:
    """Represents a city map as a graph with distances between locations."""
    
    def __init__(self):
        self.graph = defaultdict(list)
        self.coordinates = {}
        self.visited_count = {}  # Track exploration count for each algorithm
    
    def add_edge(self, city1, city2, distance):
        """Add bidirectional edge between two cities."""
        self.graph[city1].append((city2, distance))
        self.graph[city2].append((city1, distance))
    
    def set_coordinates(self, city, x, y):
        """Set 2D coordinates for heuristic calculations."""
        self.coordinates[city] = (x, y)
    
    def heuristic(self, city1, city2):
        """Euclidean distance heuristic for A* and Greedy search."""
        if city1 not in self.coordinates or city2 not in self.coordinates:
            return 0
        x1, y1 = self.coordinates[city1]
        x2, y2 = self.coordinates[city2]
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def get_neighbors(self, city):
        """Get neighboring cities and distances."""
        return self.graph.get(city, [])
    
    def city_exists(self, city):
        """Check if city exists in graph."""
        return city in self.graph or city in self.coordinates


def a_star_search(graph, start, goal):
    """
    A* Search Algorithm
    Uses both actual cost (g) and heuristic estimate (h)
    """
    if not graph.city_exists(start):
        return None, 0, [], "Start city not found"
    if not graph.city_exists(goal):
        return None, 0, [], "Goal city not found"
    
    if start == goal:
        return 0, 1, [start], "Start and goal are the same"
    
    # Priority queue: (f_score, counter, current_city, path, g_score)
    counter = 0
    open_set = [(0, counter, start, [start], 0)]
    closed_set = set()
    explored_count = 0
    
    while open_set:
        f_score, _, current, path, g_score = heapq.heappop(open_set)
        explored_count += 1
        
        if current == goal:
            return g_score, explored_count, path, "Path found"
        
        if current in closed_set:
            continue
        
        closed_set.add(current)
        
        for neighbor, distance in graph.get_neighbors(current):
            if neighbor not in closed_set:
                new_g = g_score + distance
                h_score = graph.heuristic(neighbor, goal)
                f_score_new = new_g + h_score
                counter += 1
                heapq.heappush(open_set, (f_score_new, counter, neighbor, path + [neighbor], new_g))
    
    return None, explored_count, [], "No path found"


def bfs_search(graph, start, goal):
    """
    Breadth-First Search (BFS) Algorithm
    Explores level by level, guarantees shortest path in unweighted graphs
    """
    if not graph.city_exists(start):
        return None, 0, [], "Start city not found"
    if not graph.city_exists(goal):
        return None, 0, [], "Goal city not found"
    
    if start == goal:
        return 0, 1, [start], "Start and goal are the same"
    
    queue = deque([(start, [start], 0)])
    visited = {start}
    explored_count = 0
    
    while queue:
        current, path, cost = queue.popleft()
        explored_count += 1
        
        for neighbor, distance in graph.get_neighbors(current):
            if neighbor == goal:
                return cost + distance, explored_count + 1, path + [neighbor], "Path found"
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor], cost + distance))
    
    return None, explored_count, [], "No path found"


def greedy_search(graph, start, goal):
    """
    Greedy Search Algorithm
    Only uses heuristic (h), faster but not guaranteed to find optimal path
    """
    if not graph.city_exists(start):
        return None, 0, [], "Start city not found"
    if not graph.city_exists(goal):
        return None, 0, [], "Goal city not found"
    
    if start == goal:
        return 0, 1, [start], "Start and goal are the same"
    
    # Priority queue: (h_score, counter, current_city, path, cost)
    counter = 0
    open_set = [(graph.heuristic(start, goal), counter, start, [start], 0)]
    closed_set = set()
    explored_count = 0
    
    while open_set:
        h_score, _, current, path, cost = heapq.heappop(open_set)
        explored_count += 1
        
        if current == goal:
            return cost, explored_count, path, "Path found"
        
        if current in closed_set:
            continue
        
        closed_set.add(current)
        
        for neighbor, distance in graph.get_neighbors(current):
            if neighbor not in closed_set:
                new_cost = cost + distance
                h_score_new = graph.heuristic(neighbor, goal)
                counter += 1
                heapq.heappush(open_set, (h_score_new, counter, neighbor, path + [neighbor], new_cost))
    
    return None, explored_count, [], "No path found"


def create_addis_ababa_map():
    """Create a sample map of Addis Ababa with major landmarks."""
    graph = CityGraph()
    
    # Define edges (city1, city2, distance in km)
    edges = [
        ("Arada", "Lideta", 2),
        ("Arada", "Kirkos", 1.5),
        ("Lideta", "Addis Ketema", 2),
        ("Lideta", "Kirkos", 1),
        ("Kirkos", "Nifas Silk Lafto", 3),
        ("Addis Ketema", "Akaki Kality", 4),
        ("Nifas Silk Lafto", "Yeka", 2.5),
        ("Akaki Kality", "Yeka", 3),
        ("Yeka", "Bole", 2),
        ("Bole", "Kolfe Keranio", 3),
        ("Kolfe Keranio", "Gulele", 2.5),
        ("Gulele", "Arada", 3),
    ]
    
    for city1, city2, distance in edges:
        graph.add_edge(city1, city2, distance)
    
    # Set approximate coordinates for heuristic
    coordinates = {
        "Arada": (10, 10),
        "Lideta": (12, 8),
        "Kirkos": (11, 9),
        "Addis Ketema": (13, 6),
        "Nifas Silk Lafto": (14, 11),
        "Yeka": (15, 9),
        "Akaki Kality": (14, 5),
        "Bole": (16, 7),
        "Kolfe Keranio": (12, 4),
        "Gulele": (8, 6),
    }
    
    for city, (x, y) in coordinates.items():
        graph.set_coordinates(city, x, y)
    
    return graph


def print_result(algorithm_name, cost, explored, path, message):
    """Pretty print results for an algorithm."""
    print(f"\n{'='*60}")
    print(f"Algorithm: {algorithm_name}")
    print(f"{'='*60}")
    print(f"Status: {message}")
    if path:
        print(f"Path: {' → '.join(path)}")
        print(f"Total Cost: {cost} km")
        print(f"Nodes Explored: {explored}")
        print(f"Path Length: {len(path)} cities")
    else:
        print(f"Result: No path found")


def validate_input(start, goal, graph):
    """Validate user input for edge cases."""
    errors = []
    
    # Check if cities exist
    if not graph.city_exists(start):
        errors.append(f"Error: Start city '{start}' not found in map")
    if not graph.city_exists(goal):
        errors.append(f"Error: Goal city '{goal}' not found in map")
    
    # Check if start and goal are the same
    if start == goal and graph.city_exists(start):
        errors.append(f"Note: Start and goal are the same city '{start}'")
    
    return errors


def print_available_cities(graph):
    """Print all available cities."""
    cities = sorted(set(list(graph.graph.keys()) + list(graph.coordinates.keys())))
    print("\nAvailable cities in map:")
    for i, city in enumerate(cities, 1):
        print(f"  {i}. {city}")


def main():
    print("="*60)
    print("PATHFINDING ALGORITHMS - CITY MAP NAVIGATION")
    print("="*60)
    
    # Create the city map
    graph = create_addis_ababa_map()
    
    # Display available cities
    print_available_cities(graph)
    
    # Get user input with validation loop
    while True:
        start = input("\nEnter start city (or 'quit' to exit): ").strip().title()
        if start.lower() == 'quit':
            print("Exiting program.")
            break
        
        goal = input("Enter goal city: ").strip().title()
        
        # Validate input
        errors = validate_input(start, goal, graph)
        if errors:
            for error in errors:
                print(error)
            continue
        
        print("\n" + "="*60)
        print("Running all three algorithms...")
        print("="*60)
        
        # Run all three algorithms
        cost_a, explored_a, path_a, msg_a = a_star_search(graph, start, goal)
        cost_b, explored_b, path_b, msg_b = bfs_search(graph, start, goal)
        cost_g, explored_g, path_g, msg_g = greedy_search(graph, start, goal)
        
        # Print results
        print_result("A* SEARCH", cost_a, explored_a, path_a, msg_a)
        print_result("BFS (Breadth-First Search)", cost_b, explored_b, path_b, msg_b)
        print_result("GREEDY SEARCH", cost_g, explored_g, path_g, msg_g)
        
        # Comparison
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        if path_a and path_b and path_g:
            print(f"A* explored {explored_a} nodes | Cost: {cost_a}")
            print(f"BFS explored {explored_b} nodes | Cost: {cost_b}")
            print(f"Greedy explored {explored_g} nodes | Cost: {cost_g}")
            if cost_a <= cost_b and cost_a <= cost_g:
                print("\nA* found the optimal path (lowest cost)")
        
        another = input("\nSearch for another path? (yes/no): ").strip().lower()
        if another != 'yes' and another != 'y':
            print("Thank you for using the pathfinding system!")
            break


if __name__ == "__main__":
    main()
