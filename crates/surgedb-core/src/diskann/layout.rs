//! Disk layout definitions for DiskANN index
//!
//! Defines the binary format for the graph index stored on disk.

/// Magic bytes for DiskANN graph file
pub const GRAPH_MAGIC: &[u8; 4] = b"DANN";

/// Version 1
pub const GRAPH_VERSION: u32 = 1;

/// File header (fixed size 4096 bytes)
#[repr(C)]
pub struct GraphHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub num_nodes: u64,
    pub max_degree: u32,
    pub entry_point: u32,
    pub _padding: [u8; 4076],
}

impl Default for GraphHeader {
    fn default() -> Self {
        Self {
            magic: *GRAPH_MAGIC,
            version: GRAPH_VERSION,
            num_nodes: 0,
            max_degree: 32,
            entry_point: 0,
            _padding: [0; 4076],
        }
    }
}

/// Calculate the size of a node record in bytes
pub fn node_size_bytes(max_degree: u32) -> usize {
    // Neighbor count (4 bytes) + Neighbors (4 * R bytes)
    // Align to 4 bytes (naturally aligned)
    4 + (max_degree as usize * 4)
}

/// Helper to serialize a node's adjacency list
pub fn serialize_node(neighbors: &[u32], max_degree: usize) -> Vec<u8> {
    let mut buffer = Vec::with_capacity(4 + max_degree * 4);

    // Write count
    let count = neighbors.len().min(max_degree) as u32;
    buffer.extend_from_slice(&count.to_le_bytes());

    // Write neighbors
    for &neighbor in neighbors.iter().take(max_degree) {
        buffer.extend_from_slice(&neighbor.to_le_bytes());
    }

    // Pad if fewer neighbors than max_degree
    for _ in neighbors.len()..max_degree {
        buffer.extend_from_slice(&0u32.to_le_bytes());
    }

    buffer
}
