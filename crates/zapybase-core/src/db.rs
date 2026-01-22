use crate::types::VectorId;
use crate::{
    Config, Error, QuantizationType, QuantizedConfig, QuantizedVectorDb, Result, VectorDb,
};
use parking_lot::RwLock;
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize)]
pub struct CollectionStats {
    pub vector_count: usize,
    pub memory_usage_bytes: usize,
    pub quantization: String,
    pub dimensions: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct DatabaseStats {
    pub collections: HashMap<String, CollectionStats>,
    pub total_vectors: usize,
    pub total_memory_bytes: usize,
}

/// Enum representing either a standard or quantized collection
pub enum Collection {
    Standard(Arc<RwLock<VectorDb>>),
    Quantized(Arc<RwLock<QuantizedVectorDb>>),
}

impl Collection {
    pub fn insert(&self, id: String, vector: &[f32], metadata: Option<Value>) -> Result<()> {
        match self {
            Collection::Standard(db) => db.write().insert(id, vector, metadata),
            Collection::Quantized(db) => db.write().insert(id, vector, metadata),
        }
    }

    pub fn upsert(&self, id: String, vector: &[f32], metadata: Option<Value>) -> Result<()> {
        match self {
            Collection::Standard(db) => db.write().upsert(id, vector, metadata),
            Collection::Quantized(db) => db.write().upsert(id, vector, metadata),
        }
    }

    pub fn get(&self, id: &str) -> Result<Option<(Vec<f32>, Option<Value>)>> {
        match self {
            Collection::Standard(db) => db.read().get(id),
            Collection::Quantized(db) => db.read().get(id),
        }
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(VectorId, f32, Option<Value>)>> {
        match self {
            Collection::Standard(db) => db.read().search(query, k),
            Collection::Quantized(db) => db.read().search(query, k),
        }
    }

    pub fn list(&self, offset: usize, limit: usize) -> Vec<VectorId> {
        match self {
            Collection::Standard(db) => db.read().list(offset, limit),
            Collection::Quantized(db) => db.read().list(offset, limit),
        }
    }

    pub fn stats(&self) -> CollectionStats {
        match self {
            Collection::Standard(db) => {
                let db = db.read();
                CollectionStats {
                    vector_count: db.len(),
                    memory_usage_bytes: db.memory_usage(),
                    quantization: "None".to_string(),
                    dimensions: db.config().dimensions,
                }
            }
            Collection::Quantized(db) => {
                let db = db.read();
                CollectionStats {
                    vector_count: db.len(),
                    memory_usage_bytes: db.memory_usage(),
                    quantization: format!("{:?}", db.config().quantization),
                    dimensions: db.config().dimensions,
                }
            }
        }
    }
}

/// Database manages multiple vector collections
pub struct Database {
    collections: RwLock<HashMap<String, Collection>>,
}

impl Default for Database {
    fn default() -> Self {
        Self::new()
    }
}

impl Database {
    /// Create a new empty database
    pub fn new() -> Self {
        Self {
            collections: RwLock::new(HashMap::new()),
        }
    }

    /// Create a new collection with the given configuration
    pub fn create_collection(&self, name: &str, config: Config) -> Result<()> {
        let mut collections = self.collections.write();
        if collections.contains_key(name) {
            return Err(Error::DuplicateCollection(name.to_string()));
        }

        let collection = if config.quantization == QuantizationType::None {
            let db = VectorDb::new(config)?;
            Collection::Standard(Arc::new(RwLock::new(db)))
        } else {
            // Convert Config to QuantizedConfig
            // Note: Config has `max_vectors` which QuantizedConfig doesn't,
            // and QuantizedConfig has `keep_originals` and `rerank_multiplier` which Config doesn't.
            // We'll use defaults for now or could extend Config.
            // For now assuming keep_originals=false and default rerank
            let q_config = QuantizedConfig {
                dimensions: config.dimensions,
                distance_metric: config.distance_metric,
                hnsw: config.hnsw,
                quantization: config.quantization,
                keep_originals: false, // Default behavior
                rerank_multiplier: 3,
            };
            let db = QuantizedVectorDb::new(q_config)?;
            Collection::Quantized(Arc::new(RwLock::new(db)))
        };

        collections.insert(name.to_string(), collection);
        Ok(())
    }

    /// Delete a collection
    pub fn delete_collection(&self, name: &str) -> Result<()> {
        let mut collections = self.collections.write();
        if collections.remove(name).is_none() {
            return Err(Error::CollectionNotFound(name.to_string()));
        }
        Ok(())
    }

    /// Get a collection by name
    /// Returns a reference-counted handle to the collection wrapper
    // Note: We can't return Arc<RwLock<VectorDb>> anymore because it might be Quantized.
    // We'll return a clone of the Collection enum which holds the Arcs.
    // However, Collection itself isn't Clone currently. Let's make it Clone.
    pub fn get_collection(&self, name: &str) -> Result<Collection> {
        let collections = self.collections.read();
        collections
            .get(name)
            .cloned()
            .ok_or_else(|| Error::CollectionNotFound(name.to_string()))
    }

    /// List all collection names
    pub fn list_collections(&self) -> Vec<String> {
        self.collections.read().keys().cloned().collect()
    }

    /// Get statistics for the database
    pub fn get_stats(&self) -> DatabaseStats {
        let collections = self.collections.read();
        let mut stats_map = HashMap::new();
        let mut total_vectors = 0;
        let mut total_memory = 0;

        for (name, collection) in collections.iter() {
            let stats = collection.stats();
            total_vectors += stats.vector_count;
            total_memory += stats.memory_usage_bytes;
            stats_map.insert(name.clone(), stats);
        }

        DatabaseStats {
            collections: stats_map,
            total_vectors,
            total_memory_bytes: total_memory,
        }
    }
}

// Make Collection cloneable (cheap, just cloning Arcs)
impl Clone for Collection {
    fn clone(&self) -> Self {
        match self {
            Collection::Standard(db) => Collection::Standard(db.clone()),
            Collection::Quantized(db) => Collection::Quantized(db.clone()),
        }
    }
}
