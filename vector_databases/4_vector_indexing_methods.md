# Vector Indexing Methods: Deep Dive

## Understanding Vector Indexes

### 1. Exact Search Methods

#### Brute Force (Exhaustive Search)
```
Algorithm:
1. For each query vector q
2.     For each database vector v
3.         Calculate distance(q, v)
4.     Sort distances
5.     Return k nearest neighbors

Complexity:
- Time: O(nd) where n = number of vectors, d = dimensions
- Space: O(n)
```

#### KD-Trees
```
Structure:
          [Point split on x]
         /                 \
[Point split on y]   [Point split on y]
     /        \          /        \
   ...        ...      ...       ...

Algorithm:
1. Build tree by recursively splitting on dimensions
2. Search by traversing tree, pruning branches
3. Backtrack if necessary to find true nearest neighbors

Complexity:
- Build: O(n log n)
- Query: O(log n) best case, O(n) worst case
- Space: O(n)
```

### 2. Approximate Search Methods

#### Locality-Sensitive Hashing (LSH)
```
Algorithm:
1. Define k hash functions that map similar vectors to same buckets
2. Hash all database vectors using these functions
3. For query vector:
   - Hash using same functions
   - Retrieve vectors from matching buckets
   - Compute exact distances on this subset

Implementation:
```python
import numpy as np
from typing import List, Tuple

class LSHIndex:
    def __init__(
        self,
        dim: int,
        num_tables: int,
        num_bits: int
    ):
        self.dim = dim
        self.num_tables = num_tables
        self.num_bits = num_bits
        self.tables = []
        self.random_vectors = []
        
        # Initialize hash tables
        for _ in range(num_tables):
            self.tables.append({})
            self.random_vectors.append(
                np.random.randn(num_bits, dim)
            )
    
    def _hash_vector(
        self,
        vector: np.ndarray,
        random_vectors: np.ndarray
    ) -> Tuple[int]:
        """
        Hash a vector to a bucket using random projections
        """
        projections = np.dot(random_vectors, vector)
        return tuple(projections > 0)
    
    def add_vector(
        self,
        vector: np.ndarray,
        vector_id: int
    ):
        """
        Add a vector to all hash tables
        """
        for table_idx, table in enumerate(self.tables):
            hash_value = self._hash_vector(
                vector,
                self.random_vectors[table_idx]
            )
            if hash_value not in table:
                table[hash_value] = []
            table[hash_value].append((vector_id, vector))
    
    def query(
        self,
        query_vector: np.ndarray,
        k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Find approximate nearest neighbors
        """
        candidates = set()
        
        # Collect candidates from all tables
        for table_idx, table in enumerate(self.tables):
            hash_value = self._hash_vector(
                query_vector,
                self.random_vectors[table_idx]
            )
            if hash_value in table:
                candidates.update(table[hash_value])
        
        # Calculate exact distances for candidates
        distances = []
        for vector_id, vector in candidates:
            distance = np.linalg.norm(
                query_vector - vector
            )
            distances.append((vector_id, distance))
        
        # Return k nearest neighbors
        return sorted(
            distances,
            key=lambda x: x[1]
        )[:k]
```

#### Hierarchical Navigable Small World (HNSW)
```
Structure:
Layer N:   [Sparse Graph]
Layer N-1: [Denser Graph]
...
Layer 0:   [Densest Graph]

Algorithm:
1. Build multiple layers of graphs
2. Upper layers: sparse connections
3. Lower layers: denser connections
4. Search:
   - Start at top layer
   - Find approximate nearest neighbor
   - Use as entry point for lower layer
   - Repeat until bottom layer

Implementation:
```python
import numpy as np
from typing import List, Dict, Set
import heapq

class HNSWNode:
    def __init__(self, id: int, vector: np.ndarray):
        self.id = id
        self.vector = vector
        self.neighbors: Dict[int, Set[int]] = {}  # layer -> neighbor_ids

class HNSWIndex:
    def __init__(
        self,
        dim: int,
        M: int = 16,  # Max neighbors per node
        ef_construction: int = 200,
        num_layers: int = 4
    ):
        self.dim = dim
        self.M = M
        self.ef_construction = ef_construction
        self.num_layers = num_layers
        self.nodes: Dict[int, HNSWNode] = {}
        self.entry_point = None
        
    def _distance(
        self,
        v1: np.ndarray,
        v2: np.ndarray
    ) -> float:
        return np.linalg.norm(v1 - v2)
    
    def _search_layer(
        self,
        query: np.ndarray,
        entry_point: int,
        ef: int,
        layer: int
    ) -> List[int]:
        """
        Search for nearest neighbors in a single layer
        """
        visited = {entry_point}
        candidates = [(
            self._distance(
                query,
                self.nodes[entry_point].vector
            ),
            entry_point
        )]
        heapq.heapify(candidates)
        
        while candidates:
            _, current = heapq.heappop(candidates)
            
            # Check neighbors
            for neighbor_id in self.nodes[current].neighbors[layer]:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    dist = self._distance(
                        query,
                        self.nodes[neighbor_id].vector
                    )
                    heapq.heappush(
                        candidates,
                        (dist, neighbor_id)
                    )
        
        return sorted(
            list(visited),
            key=lambda x: self._distance(
                query,
                self.nodes[x].vector
            )
        )[:ef]
    
    def add_vector(
        self,
        vector: np.ndarray,
        vector_id: int
    ):
        """
        Add a vector to the index
        """
        node = HNSWNode(vector_id, vector)
        self.nodes[vector_id] = node
        
        # Determine maximum layer for this node
        max_layer = 0
        while np.random.random() < 0.5 and max_layer < self.num_layers - 1:
            max_layer += 1
        
        # Initialize empty neighbor sets for each layer
        for layer in range(max_layer + 1):
            node.neighbors[layer] = set()
        
        if not self.entry_point:
            self.entry_point = vector_id
            return
        
        # Connect the node to the graph
        entry_point = self.entry_point
        for layer in range(max_layer, -1, -1):
            # Find nearest neighbors at this layer
            neighbors = self._search_layer(
                vector,
                entry_point,
                self.ef_construction,
                layer
            )
            
            # Connect mutual neighbors
            for neighbor_id in neighbors[:self.M]:
                node.neighbors[layer].add(neighbor_id)
                self.nodes[neighbor_id].neighbors[layer].add(vector_id)
            
            entry_point = neighbors[0]
        
        # Update entry point if necessary
        if max_layer > 0:
            self.entry_point = vector_id
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Search for k nearest neighbors
        """
        if not self.entry_point:
            return []
        
        current_layer = self.num_layers - 1
        entry_point = self.entry_point
        
        # Traverse from top to bottom layer
        while current_layer >= 0:
            neighbors = self._search_layer(
                query,
                entry_point,
                1 if current_layer > 0 else k,
                current_layer
            )
            entry_point = neighbors[0]
            current_layer -= 1
        
        # Return k nearest neighbors with distances
        results = [
            (
                n,
                self._distance(
                    query,
                    self.nodes[n].vector
                )
            )
            for n in neighbors
        ]
        return sorted(
            results,
            key=lambda x: x[1]
        )[:k]
```

#### IVF (Inverted File Index)
```
Structure:
1. Coarse quantizer (k-means centroids)
2. Inverted lists for each centroid
3. Vector residuals stored in lists

Algorithm:
1. Training:
   - Run k-means to get centroids
   - Assign vectors to nearest centroids
   - Store residuals in inverted lists

2. Search:
   - Find nearest centroids to query
   - Search corresponding inverted lists
   - Compute exact distances using residuals

Implementation:
```python
import numpy as np
from typing import List, Dict, Tuple
from sklearn.cluster import KMeans

class IVFIndex:
    def __init__(
        self,
        dim: int,
        n_lists: int = 100,
        nprobe: int = 10
    ):
        self.dim = dim
        self.n_lists = n_lists
        self.nprobe = nprobe
        self.quantizer = KMeans(
            n_clusters=n_lists,
            n_init=1
        )
        self.inverted_lists: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        
    def train(self, vectors: np.ndarray):
        """
        Train the coarse quantizer
        """
        self.quantizer.fit(vectors)
        
        # Initialize inverted lists
        for i in range(self.n_lists):
            self.inverted_lists[i] = []
    
    def add_vector(
        self,
        vector: np.ndarray,
        vector_id: int
    ):
        """
        Add a vector to the index
        """
        # Find nearest centroid
        centroid_id = self.quantizer.predict([vector])[0]
        
        # Calculate and store residual
        centroid = self.quantizer.cluster_centers_[centroid_id]
        residual = vector - centroid
        
        # Add to inverted list
        self.inverted_lists[centroid_id].append(
            (vector_id, residual)
        )
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Search for k nearest neighbors
        """
        # Find nprobe nearest centroids
        distances = [
            (
                i,
                np.linalg.norm(query - centroid)
            )
            for i, centroid in enumerate(
                self.quantizer.cluster_centers_
            )
        ]
        nearest_centroids = sorted(
            distances,
            key=lambda x: x[1]
        )[:self.nprobe]
        
        # Search in selected inverted lists
        candidates = []
        for centroid_id, _ in nearest_centroids:
            centroid = self.quantizer.cluster_centers_[centroid_id]
            
            # Calculate distances to vectors in this list
            for vector_id, residual in self.inverted_lists[centroid_id]:
                # Reconstruct original vector
                vector = centroid + residual
                distance = np.linalg.norm(query - vector)
                candidates.append((vector_id, distance))
        
        # Return k nearest neighbors
        return sorted(
            candidates,
            key=lambda x: x[1]
        )[:k]
```

## Comparison of Methods

### Performance Characteristics

| Method | Build Time | Query Time | Memory | Accuracy | Best For |
|--------|------------|------------|---------|----------|-----------|
| Brute Force | O(1) | O(nd) | O(n) | 100% | Small datasets |
| KD-Tree | O(n log n) | O(log n) - O(n) | O(n) | 100% | Low dimensions |
| LSH | O(n) | O(1) - O(n) | O(n) | Approximate | High dimensions |
| HNSW | O(n log n) | O(log n) | O(n log n) | Near 100% | General purpose |
| IVF | O(n) | O(n/k) | O(n) | Approximate | Large datasets |

### Use Case Recommendations

1. **Small Datasets (< 10K vectors)**
   - Use Brute Force or KD-Trees
   - Prioritize accuracy over speed
   - Simple implementation

2. **Medium Datasets (10K - 1M vectors)**
   - Use HNSW
   - Good balance of speed and accuracy
   - Reasonable memory usage

3. **Large Datasets (> 1M vectors)**
   - Use IVF or LSH
   - Can handle scale
   - Trade accuracy for speed

4. **High Dimensional Data**
   - Avoid KD-Trees
   - Prefer HNSW or LSH
   - Consider dimensionality reduction

5. **Real-time Applications**
   - Use HNSW
   - Fast query times
   - Good accuracy

6. **Batch Processing**
   - Use IVF
   - Good throughput
   - Memory efficient

## Optimization Strategies

### 1. Data Preprocessing
- Normalize vectors
- Reduce dimensions (PCA, t-SNE)
- Handle sparse data efficiently

### 2. Index Parameters
```python
# HNSW optimization example
def optimize_hnsw_params(
    data_size: int,
    dim: int
) -> Dict:
    """
    Calculate optimal HNSW parameters
    """
    # M (max neighbors) increases with log of data size
    M = min(64, max(16, int(np.log2(data_size))))
    
    # ef_construction increases with data size
    ef_construction = min(800, max(100, int(data_size / 1000)))
    
    # Number of layers based on data size
    num_layers = max(2, int(np.log2(data_size) / 2))
    
    return {
        "M": M,
        "ef_construction": ef_construction,
        "num_layers": num_layers
    }
```

### 3. Memory Management
```python
def estimate_memory_usage(
    num_vectors: int,
    dim: int,
    index_type: str
) -> Dict:
    """
    Estimate memory requirements
    """
    vector_size = dim * 4  # 4 bytes per float
    base_memory = num_vectors * vector_size
    
    if index_type == "hnsw":
        # HNSW typically uses 50-100 bytes per element extra
        index_memory = num_vectors * 75
    elif index_type == "ivf":
        # IVF overhead is mainly from centroids
        index_memory = int(np.sqrt(num_vectors)) * vector_size
    else:  # brute force
        index_memory = 0
    
    return {
        "base_memory_mb": base_memory / (1024 * 1024),
        "index_memory_mb": index_memory / (1024 * 1024),
        "total_memory_mb": (base_memory + index_memory) / (1024 * 1024)
    }
```

### 4. Query Optimization
```python
def optimize_query_strategy(
    num_vectors: int,
    query_latency_req: float,
    accuracy_req: float
) -> Dict:
    """
    Determine optimal query strategy
    """
    if query_latency_req < 0.001:  # sub-millisecond
        return {
            "index_type": "hnsw",
            "params": {
                "ef_search": 50,
                "use_cache": True
            }
        }
    elif accuracy_req > 0.99:  # high accuracy
        return {
            "index_type": "ivf",
            "params": {
                "nprobe": int(np.sqrt(num_vectors)),
                "use_precomputed": True
            }
        }
    else:  # balanced
        return {
            "index_type": "hnsw",
            "params": {
                "ef_search": 100,
                "use_cache": False
            }
        }
```