//! Core types for SurgeDB

use serde::{Deserialize, Serialize};

/// A vector represented as a slice of f32 values
pub type Vector = [f32];

/// External vector identifier (user-facing)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VectorId(String);

impl VectorId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for VectorId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for VectorId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl std::fmt::Display for VectorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Internal vector identifier (for indexing)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InternalId(pub(crate) u32);

impl InternalId {
    pub fn as_usize(&self) -> usize {
        self.0 as usize
    }
}

impl From<usize> for InternalId {
    fn from(id: usize) -> Self {
        Self(id as u32)
    }
}
