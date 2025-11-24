import heapq
from collections import deque, defaultdict
import math

class CityGraph:
    
    def __init__(self):
        self.graph = defaultdict(list)
        self.coordinates = {}
    
    def add_edge(self, city1, city2, distance):
        """Add bidirectional edge between two cities."""
        self.graph[city1].append((city2, distance))
        self.graph[city2].append((city1, distance))
    
    def set_coordinates(self, city, x, y):
        """Set 2D coordinates for heuristic calculations."""
        self.coordinates[city] = (x, y)
    
    def heuristic(self, city1, city2):
        """
        Haversine distance
        """
        if city1 not in self.coordinates or city2 not in self.coordinates:
            return 0
        
        lat1, lon1 = self.coordinates[city1]
        lat2, lon2 = self.coordinates[city2]
        
        R = 6371  # Earth's radius in km
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def get_neighbors(self, city):
        """Get neighboring cities and distances."""
        return self.graph.get(city, [])
    
    def city_exists(self, city):
        return city in self.graph or city in self.coordinates


def a_star_search(graph, start, goal, avoid_cities=None, max_cost=None):
    """
    Constraints:
    1. avoid_cities: Set of cities to skip
    2. max_cost: Maximum allowable path cost
    """
    if avoid_cities is None: avoid_cities = set()
    
    if not graph.city_exists(start):
        return None, 0, [], set(), set(), "Start city not found"
    if not graph.city_exists(goal):
        return None, 0, [], set(), set(), "Goal city not found"
    
    if start == goal:
        return 0, 1, [start], {start}, {start}, "Start and goal are the same"
    
    if start in avoid_cities:
        return None, 0, [], set(), set(), "Start city is in avoid list"
    if goal in avoid_cities:
        return None, 0, [], set(), set(), "Goal city is in avoid list"

    # Priority queue: (f_score, counter, current_city, path, g_score)
    counter = 0
    open_set = [(0, counter, start, [start], 0)]
    closed_set = set()
    explored_count = 0
    
    while open_set:
        f_score, _, current, path, g_score = heapq.heappop(open_set)
        
        if max_cost is not None and g_score > max_cost:
            continue

        explored_count += 1
        
        if current == goal:
            return g_score, explored_count, path, set([n for _, _, n, _, _ in open_set]), closed_set, "Path found"
        
        if current in closed_set:
            continue
        
        closed_set.add(current)
        
        for neighbor, distance in graph.get_neighbors(current):
            if neighbor not in closed_set and neighbor not in avoid_cities:
                new_g = g_score + distance
                if max_cost is not None and new_g > max_cost:
                    continue
                    
                h_score = graph.heuristic(neighbor, goal)
                f_score_new = new_g + h_score
                counter += 1
                heapq.heappush(open_set, (f_score_new, counter, neighbor, path + [neighbor], new_g))
    
    return None, explored_count, [], set(), closed_set, "No path found within constraints"


def dfs_search(graph, start, goal, avoid_cities=None, max_cost=None):
    """
    Depth-First Search (DFS) Algorithm with constraints.
    """
    if avoid_cities is None: avoid_cities = set()

    if not graph.city_exists(start):
        return None, 0, [], set(), set(), "Start city not found"
    if not graph.city_exists(goal):
        return None, 0, [], set(), set(), "Goal city not found"
    
    if start == goal:
        return 0, 1, [start], {start}, {start}, "Start and goal are the same"

    if start in avoid_cities or goal in avoid_cities:
        return None, 0, [], set(), set(), "Start or Goal city is in avoid list"
    
    stack = [(start, [start], 0)]
    visited = {start}
    explored_count = 0
    open_set_nodes = {start}
    closed_set = set()
    
    while stack:
        current, path, cost = stack.pop()
        
        if max_cost is not None and cost > max_cost:
            continue

        explored_count += 1
        
        if current == goal:
            closed_set.add(current)
            return cost, explored_count, path, open_set_nodes, closed_set, "Path found"
        
        closed_set.add(current)
        open_set_nodes.discard(current)
        
        for neighbor, distance in graph.get_neighbors(current):
            if neighbor not in visited and neighbor not in avoid_cities:
                new_cost = cost + distance
                if max_cost is not None and new_cost > max_cost:
                    continue

                visited.add(neighbor)
                open_set_nodes.add(neighbor)
                stack.append((neighbor, path + [neighbor], new_cost))
    
    return None, explored_count, [], open_set_nodes, closed_set, "No path found within constraints"


def greedy_search(graph, start, goal, avoid_cities=None, max_cost=None):
    """
    Greedy Search Algorithm with constraints.
    """
    if avoid_cities is None: avoid_cities = set()

    if not graph.city_exists(start):
        return None, 0, [], set(), set(), "Start city not found"
    if not graph.city_exists(goal):
        return None, 0, [], set(), set(), "Goal city not found"
    
    if start == goal:
        return 0, 1, [start], {start}, {start}, "Start and goal are the same"

    if start in avoid_cities or goal in avoid_cities:
        return None, 0, [], set(), set(), "Start or Goal city is in avoid list"
    
    # Priority queue: (h_score, counter, current_city, path, cost)
    counter = 0
    open_set = [(graph.heuristic(start, goal), counter, start, [start], 0)]
    closed_set = set()
    explored_count = 0
    
    while open_set:
        h_score, _, current, path, cost = heapq.heappop(open_set)
        
        if max_cost is not None and cost > max_cost:
            continue

        explored_count += 1
        
        if current == goal:
            return cost, explored_count, path, set([n for _, _, n, _, _ in open_set]), closed_set, "Path found"
        
        if current in closed_set:
            continue
        
        closed_set.add(current)
        
        for neighbor, distance in graph.get_neighbors(current):
            if neighbor not in closed_set and neighbor not in avoid_cities:
                new_cost = cost + distance
                if max_cost is not None and new_cost > max_cost:
                    continue

                h_score_new = graph.heuristic(neighbor, goal)
                counter += 1
                heapq.heappush(open_set, (h_score_new, counter, neighbor, path + [neighbor], new_cost))
    
    return None, explored_count, [], set(), closed_set, "No path found within constraints"


def create_addis_ababa_map():
    """
    Subcities included (11 official subcities):
    - Arada
    - Addis Ketema
    - Kirkos
    - Lideta
    - Nifas Silk-Lafto
    - Yeka
    - Akaki Kality
    - Bole
    - Kolfe Keranio
    - Gulele
    - Lemi Kura (newest subcity)
    """
    graph = CityGraph()
    
    edges = [
        ("Arada", "Addis Ketema", 2.16),
        ("Arada", "Lideta", 1.30),
        ("Arada", "Kirkos", 0.78),
        ("Addis Ketema", "Lideta", 1.55),
        ("Lideta", "Kirkos", 1.89),
        ("Kirkos", "Nifas Silk-Lafto", 2.45),
        ("Nifas Silk-Lafto", "Yeka", 1.35),
        ("Yeka", "Bole", 2.01),
        ("Bole", "Kolfe Keranio", 2.98),
        ("Kolfe Keranio", "Gulele", 2.39),
        ("Gulele", "Arada", 2.52),
        ("Akaki Kality", "Lideta", 3.55),
        ("Akaki Kality", "Yeka", 3.08),
        ("Bole", "Akaki Kality", 3.26),
        ("Lemi Kura", "Yeka", 2.49),
        ("Lemi Kura", "Bole", 1.01),
        ("Lemi Kura", "Akaki Kality", 3.15),
    ]
    
    for city1, city2, distance in edges:
        graph.add_edge(city1, city2, distance)
    
    # Real coordinates for Addis Ababa subcities (latitude, longitude)
    # Based on actual geographic positions
    coordinates = {
        "Arada": (9.0337, 38.7517),
        "Addis Ketema": (9.0150, 38.7733),
        "Kirkos": (9.0400, 38.7583),
        "Lideta": (9.0250, 38.7650),
        "Nifas Silk-Lafto": (9.0500, 38.7850),
        "Yeka": (9.0450, 38.7950),
        "Akaki Kality": (8.9800, 38.7800),
        "Bole": (9.0200, 38.8050),
        "Kolfe Keranio": (8.9900, 38.7700),
        "Gulele": (9.0550, 38.7400),
        "Lemi Kura": (9.0400, 38.8400),
    }
    
    for city, (lat, lon) in coordinates.items():
        graph.set_coordinates(city, lon, lat)
    
    return graph


def print_result(algorithm_name, cost, explored, path, open_set, closed_set, message):
    """Pretty print results for an algorithm with open and closed sets."""
    print(f"\n{'='*70}")
    print(f"Algorithm: {algorithm_name}")
    print(f"{'='*70}")
    print(f"Status: {message}")
    if path:
        print(f"Path: {' → '.join(path)}")
        print(f"Total Cost: {cost} km")
        print(f"Nodes Explored: {explored}")
        print(f"Path Length: {len(path)} cities")
        print(f"\nOPEN SET (nodes to explore): {open_set if open_set else 'Empty'}")
        print(f"CLOSED SET (explored nodes): {closed_set}")
    else:
        print(f"Result: No path found")


def validate_input(start, goal, graph, avoid_cities=None):
    """Validate user input including constraints."""
    errors = []
    
    if not graph.city_exists(start):
        errors.append(f"Error: Start city '{start}' not found in map")
    if not graph.city_exists(goal):
        errors.append(f"Error: Goal city '{goal}' not found in map")
    
    if start == goal and graph.city_exists(start):
        errors.append(f"Note: Start and goal are the same city '{start}'")
    
    if avoid_cities:
        if start in avoid_cities:
            errors.append(f"Error: Start city '{start}' is in the avoid list")
        if goal in avoid_cities:
            errors.append(f"Error: Goal city '{goal}' is in the avoid list")
        for city in avoid_cities:
            if not graph.city_exists(city):
                errors.append(f"Warning: Avoided city '{city}' not found in map")
    
    return errors


def print_available_cities(graph):
    """Print all available cities."""
    cities = sorted(set(list(graph.graph.keys()) + list(graph.coordinates.keys())))
    print("\nAvailable cities in map:")
    for i, city in enumerate(cities, 1):
        print(f"  {i}. {city}")


def main():
    print("="*70)
    print("PATHFINDING ALGORITHMS - CITY MAP NAVIGATION")
    print("Algorithms: A* Search, DFS (Depth-First Search), Greedy Search")
    print("="*70)
    
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
        
        avoid_input = input("Enter cities to avoid (comma separated, optional): ").strip()
        avoid_cities = {c.strip().title() for c in avoid_input.split(',') if c.strip()}
        
        max_cost_input = input("Enter max path cost (optional, press Enter for none): ").strip()
        max_cost = float(max_cost_input) if max_cost_input else None
        
        # Validate input
        errors = validate_input(start, goal, graph, avoid_cities)
        if errors:
            for error in errors:
                print(error)
            continue
        
        print("\n" + "="*70)
        print(f"Running search with constraints:")
        print(f"- Avoid: {', '.join(avoid_cities) if avoid_cities else 'None'}")
        print(f"- Max Cost: {max_cost if max_cost else 'None'}")
        print("="*70)
        
        cost_a, explored_a, path_a, open_a, closed_a, msg_a = a_star_search(graph, start, goal, avoid_cities, max_cost)
        cost_d, explored_d, path_d, open_d, closed_d, msg_d = dfs_search(graph, start, goal, avoid_cities, max_cost)
        cost_g, explored_g, path_g, open_g, closed_g, msg_g = greedy_search(graph, start, goal, avoid_cities, max_cost)
        
        # Print results
        print_result("A* SEARCH", cost_a, explored_a, path_a, open_a, closed_a, msg_a)
        print_result("DFS (Depth-First Search)", cost_d, explored_d, path_d, open_d, closed_d, msg_d)
        print_result("GREEDY SEARCH", cost_g, explored_g, path_g, open_g, closed_g, msg_g)
        
        # Comparison
        print(f"\n{'='*70}")
        print("COMPARISON OF ALGORITHMS")
        print(f"{'='*70}")
        if path_a and path_d and path_g:
            print(f"A*     | Explored: {explored_a:2d} nodes | Cost: {cost_a:5.1f} km")
            print(f"DFS    | Explored: {explored_d:2d} nodes | Cost: {cost_d:5.1f} km")
            print(f"Greedy | Explored: {explored_g:2d} nodes | Cost: {cost_g:5.1f} km")
            if cost_a <= cost_d and cost_a <= cost_g:
                print("\n✓ A* found the optimal path (lowest cost)")
        
        another = input("\nSearch for another path? (yes/no): ").strip().lower()
        if another != 'yes' and another != 'y':
            print("Thank you for using the pathfinding system!")
            break


if __name__ == "__main__":
    main()
