import heapq
import pandas as pd
import numpy as np

class Graph:
    def _init_(self, vertices):
        self.V = vertices
        self.graph = [[] for _ in range(vertices)]

    def add_edge(self, u, v, w):
        self.graph[u].append((v, w))
        self.graph[v].append((u, w))  # Assuming the graph is undirected

def prim_mst(graph, start):
    min_heap = []  # (weight, distance_from_source, vertex, parent)
    visited = set()
    min_span_tree = []

    # Helper function to push into heap based on minimum weight and maximum distance
    def push_to_heap_min_weight_max_distance(node, weight, distance, parent):
        heapq.heappush(min_heap, (weight, -distance, node, parent))

    push_to_heap_min_weight_max_distance(start, 0, 0, None)

    while min_heap:
        weight, neg_distance, vertex, parent = heapq.heappop(min_heap)
        distance = -neg_distance
        if vertex not in visited:
            visited.add(vertex)
            if parent is not None:
                min_span_tree.append((parent, vertex, weight))
            for neighbor, edge_weight in graph.graph[vertex]:
                if neighbor not in visited:
                    push_to_heap_min_weight_max_distance(neighbor, edge_weight, distance + edge_weight, vertex)

    return min_span_tree

# Read the CSV file
df = pd.read_csv('part_a_input_dataset_1.csv')

# Store the order IDs and coordinates in lists
order_ids = df['order_id'].values.tolist()
coordinates = df[['lat', 'lng']].values.tolist()
depot_lat, depot_lng = df.iloc[0]['depot_lat'], df.iloc[0]['depot_lng']

# Define a function to calculate Haversine distance
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0)*2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0)*2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

# Find the origin node based on the depot latitude and longitude
min_distance = float('inf')
origin = None
for i, (lat, lng) in enumerate(coordinates):
    distance = haversine(lat, lng, depot_lat, depot_lng)
    if distance < min_distance:
        min_distance = distance
        origin = i

# Create a graph using the coordinates and distances
num_vertices = len(order_ids)
graph = Graph(num_vertices)

for i in range(num_vertices):
    for j in range(i+1, num_vertices):
        lat1, lng1 = coordinates[i]
        lat2, lng2 = coordinates[j]
        distance = haversine(lat1, lng1, lat2, lng2)
        graph.add_edge(i, j, distance)  # Add edge with weight to the graph
        # Assuming undirected graph, so adding both directions

# Choose the origin node
if origin is None:
    raise ValueError("Origin not found")

min_spanning_tree = prim_mst(graph, origin)

# Output in the desired format
output = []
for i, (parent, vertex, weight) in enumerate(min_spanning_tree):
    order_id = order_ids[vertex]
    lng, lat = coordinates[vertex]
    output.append([order_id, lng, lat, depot_lat, depot_lng, i+1])

# Print the output in the desired format
print("order_id lng lat depot_lat depot_lng dlvr_seq_num")
for entry in output:
    print(" ".join(map(str, entry)))