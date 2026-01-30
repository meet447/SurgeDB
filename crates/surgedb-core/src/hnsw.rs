//! HNSW (Hierarchical Navigable Small World) index implementation
//!
//! This is the core indexing algorithm that enables fast approximate nearest neighbor search.
//! The implementation supports:
//! - In-memory mode (fastest, for hot data)
//! - Mmap mode (for disk-resident vectors) [TODO]
//! - Hybrid mode (adaptive) [TODO]

use crate::distance::DistanceMetric;
use crate::error::{Error, Result};
use crate::filter::Filter;
use crate::storage::VectorStorageTrait;
use crate::sync::RwLock;
use crate::types::InternalId;
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

/// HNSW configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Maximum number of connections per node (M)
    pub m: usize,

    /// Maximum connections for layer 0 (M0 = 2 * M by default)
    pub m0: usize,

    /// Size of dynamic candidate list during construction (ef_construction)
    pub ef_construction: usize,

    /// Size of dynamic candidate list during search (ef_search)
    pub ef_search: usize,

    /// Normalization factor for level generation (1/ln(M))
    pub ml: f64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        let m = 16;
        Self {
            m,
            m0: m * 2,
            ef_construction: 200,
            ef_search: 100,
            ml: 1.0 / (m as f64).ln(),
        }
    }
}

impl HnswConfig {
    /// Create a config optimized for memory-constrained environments
    pub fn memory_optimized() -> Self {
        let m = 8;
        Self {
            m,
            m0: m * 2,
            ef_construction: 100,
            ef_search: 50,
            ml: 1.0 / (m as f64).ln(),
        }
    }

    /// Create a config optimized for accuracy
    pub fn accuracy_optimized() -> Self {
        let m = 32;
        Self {
            m,
            m0: m * 2,
            ef_construction: 400,
            ef_search: 200,
            ml: 1.0 / (m as f64).ln(),
        }
    }
}

/// A node in the HNSW graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswNode {
    /// The internal ID of this node (for debugging/serialization)
    pub(crate) id: InternalId,

    /// Maximum layer this node exists on
    pub(crate) max_layer: usize,

    /// Neighbors at each layer (layer -> list of neighbors)
    pub(crate) neighbors: Vec<Vec<InternalId>>,
}

impl HnswNode {
    fn new(id: InternalId, max_layer: usize) -> Self {
        Self {
            id,
            max_layer,
            neighbors: vec![Vec::new(); max_layer + 1],
        }
    }
}

/// Candidate for search with distance
#[derive(Debug, Clone, Copy)]
struct Candidate {
    id: InternalId,
    distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Max-heap candidate (for keeping track of worst candidates)
#[derive(Debug, Clone, Copy)]
struct MaxCandidate {
    id: InternalId,
    distance: f32,
}

impl PartialEq for MaxCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for MaxCandidate {}

impl PartialOrd for MaxCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MaxCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

struct SearchContext<'a> {
    query: &'a [f32],
    ef: usize,
    layer: usize,
    filter: Option<&'a Filter>,
}

/// State of the HNSW index for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswState {
    pub nodes: Vec<HnswNode>,
    pub entry_point: Option<InternalId>,
    pub max_layer: usize,
}

/// The HNSW index
pub struct HnswIndex {
    config: HnswConfig,
    distance_metric: DistanceMetric,

    /// All nodes in the graph
    nodes: RwLock<Vec<HnswNode>>,

    /// Entry point (node with highest layer)
    entry_point: RwLock<Option<InternalId>>,

    /// Maximum layer in the graph
    max_layer: RwLock<usize>,
}

impl HnswIndex {
    /// Create a new HNSW index
    pub fn new(config: HnswConfig, distance_metric: DistanceMetric) -> Self {
        Self {
            config,
            distance_metric,
            nodes: RwLock::new(Vec::new()),
            entry_point: RwLock::new(None),
            max_layer: RwLock::new(0),
        }
    }

    /// Generate a random level for a new node
    fn random_level(&self) -> usize {
        #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
        {
            let r = {
                let val = js_sys::Math::random();
                if val < f64::EPSILON {
                    f64::EPSILON
                } else {
                    val
                }
            };
            // Use js_sys::Math::log to avoid potential intrinsic issues on WASM
            let ln_val = js_sys::Math::log(r);
            (-ln_val * self.config.ml).floor() as usize
        }

        #[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
        {
            let mut rng = rand::thread_rng();
            let r: f64 = rng.gen();
            (-r.ln() * self.config.ml).floor() as usize
        }
    }

    /// Insert multiple vectors in a batch
    #[cfg(feature = "parallel")]
    pub fn insert_batch(
        &self,
        items: &[(InternalId, &[f32])],
        storage: &(impl VectorStorageTrait + Sync),
    ) -> Result<()> {
        use rayon::prelude::*; // Use inside function to avoid trait/impl conflict

        if items.is_empty() {
            return Ok(());
        }

        // Generate levels for all new nodes upfront
        let mut new_nodes_data = Vec::with_capacity(items.len());
        for (internal_id, _) in items {
            let level = self.random_level();
            new_nodes_data.push((*internal_id, level));
        }

        // Phase 1: Parallel Search for neighbors
        // We acquire a Read lock on the graph to allow parallel reading
        let search_results: Vec<_> = {
            let nodes = self.nodes.read();
            let entry_point = self.entry_point.read();
            let max_layer = *self.max_layer.read();

            if let Some(ep) = *entry_point {
                // Parallel iterator over items
                items
                    .par_iter()
                    .zip(new_nodes_data.par_iter())
                    .map(|(&(_, vector), &(_, node_level))| {
                        let mut current_ep = ep;
                        let mut neighbors_by_layer = vec![Vec::new(); node_level + 1];

                        // Traverse from top layer to node_level + 1
                        for layer in (node_level + 1..=max_layer).rev() {
                            if let Ok(next_ep) =
                                self.search_layer_single(vector, current_ep, layer, &nodes, storage)
                            {
                                current_ep = next_ep;
                            }
                        }

                        // For layers from min(node_level, max_layer) down to 0
                        let start_layer = node_level.min(max_layer);
                        for layer in (0..=start_layer).rev() {
                            let ctx = SearchContext {
                                query: vector,
                                ef: self.config.ef_construction,
                                layer,
                                filter: None,
                            };
                            if let Ok(neighbors) =
                                self.search_layer(ctx, current_ep, &nodes, storage)
                            {
                                // Select best neighbors
                                let m = if layer == 0 {
                                    self.config.m0
                                } else {
                                    self.config.m
                                };
                                let selected = self.select_neighbors(&neighbors, m, storage);
                                neighbors_by_layer[layer] = selected;

                                if !neighbors_by_layer[layer].is_empty() {
                                    current_ep = neighbors_by_layer[layer][0].id;
                                }
                            }
                        }
                        Ok::<_, crate::error::Error>(neighbors_by_layer)
                    })
                    .collect()
            } else {
                // Index is empty, nothing to search
                (0..items.len()).map(|_| Ok(Vec::new())).collect()
            }
        };

        // Phase 2: Sequential Update (Write Lock)
        // We now acquire the Write lock to actually modify the graph
        let mut nodes = self.nodes.write();
        let mut entry_point = self.entry_point.write();
        let mut max_layer = self.max_layer.write();

        // Fix type mismatch by destructuring the tuple reference correctly
        for (i, &(internal_id, level)) in new_nodes_data.iter().enumerate() {
            let mut new_node = HnswNode::new(internal_id, level);

            // If index was empty initially, the first item becomes entry point
            if entry_point.is_none() {
                *entry_point = Some(internal_id);
                *max_layer = level;
                nodes.push(new_node);
                continue;
            }

            // Assign pre-computed neighbors
            if let Ok(neighbors_by_layer) = &search_results[i] {
                for (layer, candidates) in neighbors_by_layer.iter().enumerate() {
                    if layer < new_node.neighbors.len() {
                        new_node.neighbors[layer] =
                            candidates.iter().map(|c: &Candidate| c.id).collect();
                    }
                }
            }

            nodes.push(new_node);

            // Update bidirectional connections
            if let Ok(neighbors_by_layer) = &search_results[i] {
                for (layer, candidates) in neighbors_by_layer.iter().enumerate() {
                    for neighbor in candidates {
                        let neighbor_idx = neighbor.id.as_usize();

                        if neighbor_idx < nodes.len() {
                            let neighbor_node = &mut nodes[neighbor_idx];
                            if neighbor_node.max_layer >= layer {
                                neighbor_node.neighbors[layer].push(internal_id);

                                // Prune connections if needed
                                let max_connections = if layer == 0 {
                                    self.config.m0
                                } else {
                                    self.config.m
                                };

                                if neighbor_node.neighbors[layer].len() > max_connections {
                                    if let Some(nv) = storage.get_vector_data(neighbor.id) {
                                        let mut candidates: Vec<Candidate> = neighbor_node
                                            .neighbors[layer]
                                            .iter()
                                            .filter_map(|&n_id| {
                                                storage
                                                    .distance(n_id, &nv, self.distance_metric)
                                                    .map(|dist| Candidate {
                                                        id: n_id,
                                                        distance: dist,
                                                    })
                                            })
                                            .collect();
                                        candidates.sort_by(|a, b| {
                                            a.distance
                                                .partial_cmp(&b.distance)
                                                .unwrap_or(Ordering::Equal)
                                        });
                                        neighbor_node.neighbors[layer] = candidates
                                            .into_iter()
                                            .take(max_connections)
                                            .map(|c| c.id)
                                            .collect();
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Update entry point
            if level > *max_layer {
                *max_layer = level;
                *entry_point = Some(internal_id);
            }
        }

        Ok(())
    }

    /// Insert multiple vectors in a batch (sequential version for WASM)
    #[cfg(not(feature = "parallel"))]
    pub fn insert_batch(
        &self,
        items: &[(InternalId, &[f32])],
        storage: &impl VectorStorageTrait,
    ) -> Result<()> {
        // Sequential fallback: just call insert for each item
        for &(internal_id, vector) in items {
            self.insert(internal_id, vector, storage)?;
        }
        Ok(())
    }

    /// Insert a new vector into the index
    pub fn insert(
        &self,
        internal_id: InternalId,
        vector: &[f32],
        storage: &impl VectorStorageTrait,
    ) -> Result<()> {
        let node_level = self.random_level();

        let mut nodes = self.nodes.write();
        let mut entry_point = self.entry_point.write();
        let mut max_layer = self.max_layer.write();

        // Create the new node
        let new_node = HnswNode::new(internal_id, node_level);
        nodes.push(new_node);

        // If this is the first node, set it as entry point and return
        if entry_point.is_none() {
            *entry_point = Some(internal_id);
            *max_layer = node_level;
            return Ok(());
        }

        let ep = entry_point.expect("Entry point should be Some here");
        let current_max_layer = *max_layer;

        // Search from top layer to node_level + 1, finding the closest node
        let mut current_ep = ep;
        for layer in (node_level + 1..=current_max_layer).rev() {
            current_ep = self.search_layer_single(vector, current_ep, layer, &nodes, storage)?;
        }

        // For layers from min(node_level, max_layer) down to 0, find and connect neighbors
        let start_layer = node_level.min(current_max_layer);
        for layer in (0..=start_layer).rev() {
            let ctx = SearchContext {
                query: vector,
                ef: self.config.ef_construction,
                layer,
                filter: None,
            };
            let neighbors = self.search_layer(ctx, current_ep, &nodes, storage)?;

            // Select M best neighbors using heuristic
            let m = if layer == 0 {
                self.config.m0
            } else {
                self.config.m
            };
            let selected = self.select_neighbors(&neighbors, m, storage);

            // Connect new node to selected neighbors
            let node_idx = internal_id.as_usize();
            nodes[node_idx].neighbors[layer] = selected.iter().map(|c| c.id).collect();

            // Add bidirectional connections
            for neighbor in &selected {
                let neighbor_idx = neighbor.id.as_usize();
                let neighbor_node = &mut nodes[neighbor_idx];

                if neighbor_node.max_layer >= layer {
                    neighbor_node.neighbors[layer].push(internal_id);

                    // Prune if too many connections
                    let max_connections = if layer == 0 {
                        self.config.m0
                    } else {
                        self.config.m
                    };

                    if neighbor_node.neighbors[layer].len() > max_connections {
                        // Get distances and prune
                        if let Some(nv) = storage.get_vector_data(neighbor.id) {
                            let mut candidates: Vec<Candidate> = neighbor_node.neighbors[layer]
                                .iter()
                                .filter_map(|&n_id| {
                                    storage
                                        .distance(n_id, &nv, self.distance_metric)
                                        .map(|dist| Candidate {
                                            id: n_id,
                                            distance: dist,
                                        })
                                })
                                .collect();
                            candidates.sort_by(|a, b| {
                                a.distance
                                    .partial_cmp(&b.distance)
                                    .unwrap_or(Ordering::Equal)
                            });
                            neighbor_node.neighbors[layer] = candidates
                                .into_iter()
                                .take(max_connections)
                                .map(|c| c.id)
                                .collect();
                        }
                    }
                }
            }

            if !selected.is_empty() {
                current_ep = selected[0].id;
            }
        }

        // Update entry point if new node has higher layer
        if node_level > current_max_layer {
            *entry_point = Some(internal_id);
            *max_layer = node_level;
        }

        Ok(())
    }

    /// Search for a single nearest neighbor in a layer (greedy search)
    fn search_layer_single(
        &self,
        query: &[f32],
        entry: InternalId,
        layer: usize,
        nodes: &[HnswNode],
        storage: &impl VectorStorageTrait,
    ) -> Result<InternalId> {
        let mut current = entry;
        let mut current_dist = storage
            .distance(entry, query, self.distance_metric)
            .unwrap_or(f32::MAX);

        loop {
            let node = &nodes[current.as_usize()];
            let mut changed = false;

            if node.max_layer >= layer {
                for &neighbor_id in &node.neighbors[layer] {
                    if let Some(dist) = storage.distance(neighbor_id, query, self.distance_metric) {
                        if dist < current_dist {
                            current = neighbor_id;
                            current_dist = dist;
                            changed = true;
                        }
                    }
                }
            }

            if !changed {
                break;
            }
        }

        Ok(current)
    }

    /// Search for ef nearest neighbors in a layer
    fn search_layer(
        &self,
        ctx: SearchContext,
        entry: InternalId,
        nodes: &[HnswNode],
        storage: &impl VectorStorageTrait,
    ) -> Result<Vec<Candidate>> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new(); // min-heap
        let mut results = BinaryHeap::new(); // max-heap

        let entry_dist = storage
            .distance(entry, ctx.query, self.distance_metric)
            .unwrap_or(f32::MAX);

        visited.insert(entry);
        candidates.push(Candidate {
            id: entry,
            distance: entry_dist,
        });

        // Check if entry point matches filter and is not deleted
        let entry_matches = if let Some(f) = ctx.filter {
            storage
                .get_metadata(entry)
                .map(|m| f.matches(&m))
                .unwrap_or(false)
        } else {
            true
        };
        let entry_valid = !storage.is_deleted(entry) && entry_matches;

        if entry_valid {
            results.push(MaxCandidate {
                id: entry,
                distance: entry_dist,
            });
        }

        while let Some(current) = candidates.pop() {
            // Get the furthest result
            let furthest = results.peek().map(|c| c.distance).unwrap_or(f32::MAX);

            if current.distance > furthest {
                break;
            }

            let node = &nodes[current.id.as_usize()];
            if node.max_layer >= ctx.layer {
                for &neighbor_id in &node.neighbors[ctx.layer] {
                    if visited.insert(neighbor_id) {
                        if let Some(dist) =
                            storage.distance(neighbor_id, ctx.query, self.distance_metric)
                        {
                            let furthest = results.peek().map(|c| c.distance).unwrap_or(f32::MAX);

                            if dist < furthest || results.len() < ctx.ef {
                                candidates.push(Candidate {
                                    id: neighbor_id,
                                    distance: dist,
                                });

                                // Check filter and deleted status before adding to results
                                let matches_filter = if let Some(f) = ctx.filter {
                                    storage
                                        .get_metadata(neighbor_id)
                                        .map(|m| f.matches(&m))
                                        .unwrap_or(false)
                                } else {
                                    true
                                };
                                let neighbor_valid =
                                    !storage.is_deleted(neighbor_id) && matches_filter;

                                if neighbor_valid {
                                    results.push(MaxCandidate {
                                        id: neighbor_id,
                                        distance: dist,
                                    });

                                    if results.len() > ctx.ef {
                                        results.pop();
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Convert results to sorted vector
        let mut result_vec: Vec<Candidate> = results
            .into_iter()
            .map(|c| Candidate {
                id: c.id,
                distance: c.distance,
            })
            .collect();
        result_vec.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });

        Ok(result_vec)
    }

    /// Select best neighbors using the heuristic from HNSW paper (Algorithm 4)
    fn select_neighbors(
        &self,
        candidates: &[Candidate],
        m: usize,
        storage: &impl VectorStorageTrait,
    ) -> Vec<Candidate> {
        if candidates.len() <= m {
            return candidates.to_vec();
        }

        let mut result: Vec<Candidate> = Vec::with_capacity(m);
        let mut discard = Vec::new();

        // Candidates are already sorted by distance to query (closest first)
        for &candidate in candidates {
            if result.len() >= m {
                break;
            }

            // Check if this candidate is closer to the query than to any already selected neighbor
            let mut is_closer = true;
            if let Some(candidate_vec) = storage.get_vector_data(candidate.id) {
                for &selected in &result {
                    if let Some(dist) =
                        storage.distance(selected.id, &candidate_vec, self.distance_metric)
                    {
                        if dist < candidate.distance {
                            is_closer = false;
                            break;
                        }
                    }
                }
            }

            if is_closer {
                result.push(candidate);
            } else {
                discard.push(candidate);
            }
        }

        // Fill remaining spots from discard pile if needed (to ensure connectivity)
        // This is optional but recommended to maintain M connections
        /*
        while result.len() < m && !discard.is_empty() {
            result.push(discard.remove(0));
        }
        */

        // If we filtered too aggressively and have fewer than M (but we started with > M),
        // fill up with the best discarded ones.
        if result.len() < m {
            for &candidate in &discard {
                if result.len() >= m {
                    break;
                }
                result.push(candidate);
            }
        }

        result
    }

    /// Search for k nearest neighbors
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        storage: &impl VectorStorageTrait,
        filter: Option<&Filter>,
    ) -> Result<Vec<(InternalId, f32)>> {
        let nodes = self.nodes.read();
        let entry_point = self.entry_point.read();
        let max_layer = *self.max_layer.read();

        let ep = match *entry_point {
            Some(ep) => ep,
            None => return Err(Error::EmptyIndex),
        };

        // Traverse from top layer to layer 1
        let mut current_ep = ep;
        for layer in (1..=max_layer).rev() {
            current_ep = self.search_layer_single(query, current_ep, layer, &nodes, storage)?;
        }

        // Search in layer 0 with ef_search
        let ef = self.config.ef_search.max(k);
        let ctx = SearchContext {
            query,
            ef,
            layer: 0,
            filter,
        };
        let candidates = self.search_layer(ctx, current_ep, &nodes, storage)?;

        // Return top k
        Ok(candidates
            .into_iter()
            .take(k)
            .map(|c| (c.id, c.distance))
            .collect())
    }

    /// Get the number of nodes in the index
    pub fn len(&self) -> usize {
        self.nodes.read().len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the current state of the index for serialization
    pub fn get_state(&self) -> HnswState {
        let nodes = self.nodes.read();
        let entry_point = self.entry_point.read();
        let max_layer = self.max_layer.read();

        HnswState {
            nodes: nodes.clone(),
            entry_point: *entry_point,
            max_layer: *max_layer,
        }
    }

    /// Load the index state from serialized data
    pub fn load_state(&self, state: HnswState) {
        let mut self_nodes = self.nodes.write();
        let mut self_entry_point = self.entry_point.write();
        let mut self_max_layer = self.max_layer.write();

        *self_nodes = state.nodes;
        *self_entry_point = state.entry_point;
        *self_max_layer = state.max_layer;
    }

    /// Get approximate memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let nodes = self.nodes.read();
        let mut size = nodes.capacity() * std::mem::size_of::<HnswNode>();
        for node in nodes.iter() {
            for layer in &node.neighbors {
                size += layer.capacity() * std::mem::size_of::<InternalId>();
            }
        }
        size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::VectorStorage;

    fn create_test_storage() -> VectorStorage {
        VectorStorage::new(4)
    }

    #[test]
    fn test_single_insert() {
        let config = HnswConfig::default();
        let index = HnswIndex::new(config, DistanceMetric::Cosine);
        let storage = create_test_storage();

        let id = storage
            .insert("vec1".into(), &[1.0, 0.0, 0.0, 0.0], None)
            .unwrap();
        index.insert(id, &[1.0, 0.0, 0.0, 0.0], &storage).unwrap();

        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_multiple_inserts() {
        let config = HnswConfig::default();
        let index = HnswIndex::new(config, DistanceMetric::Cosine);
        let storage = create_test_storage();

        let vectors = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
        ];

        for (i, v) in vectors.iter().enumerate() {
            let id = storage.insert(format!("vec{}", i).into(), v, None).unwrap();
            index.insert(id, v, &storage).unwrap();
        }

        assert_eq!(index.len(), 4);
    }

    #[test]
    fn test_search() {
        let config = HnswConfig::default();
        let index = HnswIndex::new(config, DistanceMetric::Cosine);
        let storage = create_test_storage();

        let vectors = vec![
            ("vec0", [1.0, 0.0, 0.0, 0.0]),
            ("vec1", [0.0, 1.0, 0.0, 0.0]),
            ("vec2", [0.0, 0.0, 1.0, 0.0]),
            ("vec3", [0.9, 0.1, 0.0, 0.0]),
            ("vec4", [0.8, 0.2, 0.0, 0.0]),
        ];

        for (name, v) in &vectors {
            let id = storage.insert((*name).into(), v, None).unwrap();
            index.insert(id, v, &storage).unwrap();
        }

        // Search for vector similar to [1, 0, 0, 0]
        let query = [1.0, 0.0, 0.0, 0.0];
        let results = index.search(&query, 3, &storage, None).unwrap();

        assert_eq!(results.len(), 3);

        // First result should be vec0 (exact match)
        let first_id = storage.get_external_id(results[0].0).unwrap();
        assert_eq!(first_id.as_str(), "vec0");
    }
}
