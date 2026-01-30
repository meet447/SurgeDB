//! Bitmap-based inverted index for fast metadata filtering
//!
//! Uses Roaring Bitmaps to store sets of internal IDs that match specific
//! metadata field-value pairs.

use crate::types::InternalId;
use roaring::RoaringBitmap;
use serde_json::Value;
use std::collections::HashMap;

/// Inverted index for metadata fields
#[derive(Default)]
pub struct BitmapIndex {
    /// field -> value_str -> bitmap
    index: HashMap<String, HashMap<String, RoaringBitmap>>,
}

impl BitmapIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Index a document's metadata
    pub fn index(&mut self, internal_id: InternalId, metadata: &Value) {
        let id = internal_id.as_u32();
        self.index_recursive(id, metadata, "");
    }

    /// Recursively index JSON fields
    fn index_recursive(&mut self, id: u32, value: &Value, prefix: &str) {
        match value {
            Value::Object(map) => {
                for (k, v) in map {
                    let key = if prefix.is_empty() {
                        k.clone()
                    } else {
                        format!("{}.{}", prefix, k)
                    };
                    self.index_recursive(id, v, &key);
                }
            }
            Value::Array(arr) => {
                // Index array elements with the same key (for "tags": ["a", "b"])
                for v in arr {
                    self.index_recursive(id, v, prefix);
                }
            }
            primitive => {
                // Index primitive value
                if !prefix.is_empty() {
                    let val_str = primitive.to_string(); // Handles numbers, bools, strings
                                                         // Remove quotes from strings for cleaner keys if desired,
                                                         // but keeping JSON string representation is safer for uniqueness

                    self.index
                        .entry(prefix.to_string())
                        .or_default()
                        .entry(val_str)
                        .or_default()
                        .insert(id);
                }
            }
        }
    }

    /// Remove a document from the index
    pub fn remove(&mut self, internal_id: InternalId, metadata: &Value) {
        let id = internal_id.as_u32();
        self.remove_recursive(id, metadata, "");
    }

    fn remove_recursive(&mut self, id: u32, value: &Value, prefix: &str) {
        match value {
            Value::Object(map) => {
                for (k, v) in map {
                    let key = if prefix.is_empty() {
                        k.clone()
                    } else {
                        format!("{}.{}", prefix, k)
                    };
                    self.remove_recursive(id, v, &key);
                }
            }
            Value::Array(arr) => {
                for v in arr {
                    self.remove_recursive(id, v, prefix);
                }
            }
            primitive => {
                if !prefix.is_empty() {
                    let val_str = primitive.to_string();
                    if let Some(values) = self.index.get_mut(prefix) {
                        if let Some(bitmap) = values.get_mut(&val_str) {
                            bitmap.remove(id);
                            // Cleanup empty bitmaps/maps could go here
                        }
                    }
                }
            }
        }
    }

    /// Execute a filter query and return matching internal IDs
    pub fn filter(&self, filter: &crate::filter::Filter) -> Option<RoaringBitmap> {
        use crate::filter::Filter;

        match filter {
            Filter::Exact(key, value) => {
                if let Some(values) = self.index.get(key) {
                    values.get(&value.to_string()).cloned()
                } else {
                    Some(RoaringBitmap::new()) // Field not found -> empty set
                }
            }
            Filter::OneOf(key, values) => {
                if let Some(field_values) = self.index.get(key) {
                    let mut result = RoaringBitmap::new();
                    for val in values {
                        if let Some(bitmap) = field_values.get(&val.to_string()) {
                            result |= bitmap;
                        }
                    }
                    Some(result)
                } else {
                    Some(RoaringBitmap::new())
                }
            }
            Filter::And(filters) => {
                let mut result: Option<RoaringBitmap> = None;
                for f in filters {
                    if let Some(bitmap) = self.filter(f) {
                        match result {
                            None => result = Some(bitmap),
                            Some(ref mut r) => *r &= bitmap,
                        }

                        // Optimization: if empty, stop
                        if result.as_ref().map(|r| r.is_empty()).unwrap_or(false) {
                            return Some(RoaringBitmap::new());
                        }
                    }
                }
                result
            }
            Filter::Or(filters) => {
                let mut result = RoaringBitmap::new();
                for f in filters {
                    if let Some(bitmap) = self.filter(f) {
                        result |= bitmap;
                    }
                }
                Some(result)
            }
            Filter::Not(_filter) => {
                // NOT is hard because we need the "universe" set (all valid IDs).
                // We typically handle NOT by post-filtering or by passing the universe explicitly.
                // For now, return None to fallback to scan-based filtering for NOT.
                None
            }
            Filter::Range { .. } | Filter::GeoRadius { .. } => {
                // Range queries on bitmaps require range-encoded bitmaps or B-trees.
                // Fallback to scan for now.
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_indexing_and_filtering() {
        let mut index = BitmapIndex::new();

        index.index(InternalId::from(1), &json!({ "tag": "A", "val": 10 }));
        index.index(InternalId::from(2), &json!({ "tag": "B", "val": 20 }));
        index.index(InternalId::from(3), &json!({ "tag": "A", "val": 20 }));

        // Exact match "tag": "A" -> {1, 3}
        let filter = crate::filter::Filter::Exact("tag".to_string(), json!("A"));
        let result = index.filter(&filter).unwrap();
        assert!(result.contains(1));
        assert!(result.contains(3));
        assert!(!result.contains(2));

        // Exact match "val": 20 -> {2, 3}
        let filter = crate::filter::Filter::Exact("val".to_string(), json!(20));
        let result = index.filter(&filter).unwrap();
        assert!(result.contains(2));
        assert!(result.contains(3));
        assert!(!result.contains(1));

        // AND -> {3}
        let filter = crate::filter::Filter::And(vec![
            crate::filter::Filter::Exact("tag".to_string(), json!("A")),
            crate::filter::Filter::Exact("val".to_string(), json!(20)),
        ]);
        let result = index.filter(&filter).unwrap();
        assert!(result.contains(3));
        assert_eq!(result.len(), 1);
    }
}
