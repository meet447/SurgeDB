use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Filter {
    /// Exact match: key == value
    Exact(String, Value),
    /// One of: key in [values]
    OneOf(String, Vec<Value>),
    /// Logical AND
    And(Vec<Filter>),
    /// Logical OR
    Or(Vec<Filter>),
    /// Logical NOT
    Not(Box<Filter>),
}

impl Filter {
    /// Check if the metadata matches the filter
    pub fn matches(&self, metadata: &Value) -> bool {
        match self {
            Filter::Exact(key, expected_value) => {
                if let Some(actual_value) = get_value_by_path(metadata, key) {
                    actual_value == expected_value
                } else {
                    false
                }
            }
            Filter::OneOf(key, allowed_values) => {
                if let Some(actual_value) = get_value_by_path(metadata, key) {
                    allowed_values.contains(actual_value)
                } else {
                    false
                }
            }
            Filter::And(filters) => filters.iter().all(|f| f.matches(metadata)),
            Filter::Or(filters) => filters.iter().any(|f| f.matches(metadata)),
            Filter::Not(filter) => !filter.matches(metadata),
        }
    }
}

/// Helper to get a value from a JSON object using a dot-notation path
fn get_value_by_path<'a>(metadata: &'a Value, path: &str) -> Option<&'a Value> {
    if path.is_empty() {
        return Some(metadata);
    }

    let mut current = metadata;
    for part in path.split('.') {
        match current {
            Value::Object(map) => {
                if let Some(next) = map.get(part) {
                    current = next;
                } else {
                    return None;
                }
            }
            _ => return None,
        }
    }
    Some(current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_exact_match() {
        let meta = json!({
            "category": "books",
            "year": 2023,
            "publisher": {
                "name": "O'Reilly",
                "location": "CA"
            }
        });

        let filter = Filter::Exact("category".to_string(), json!("books"));
        assert!(filter.matches(&meta));

        let filter_bad = Filter::Exact("category".to_string(), json!("movies"));
        assert!(!filter_bad.matches(&meta));
    }

    #[test]
    fn test_nested_path() {
        let meta = json!({
            "publisher": {
                "name": "O'Reilly",
                "location": "CA"
            }
        });

        let filter = Filter::Exact("publisher.location".to_string(), json!("CA"));
        assert!(filter.matches(&meta));
    }

    #[test]
    fn test_logical_operators() {
        let meta = json!({
            "category": "ai",
            "public": true
        });

        let filter = Filter::And(vec![
            Filter::Exact("public".to_string(), json!(true)),
            Filter::Or(vec![
                Filter::Exact("category".to_string(), json!("ai")),
                Filter::Exact("category".to_string(), json!("database")),
            ]),
        ]);

        assert!(filter.matches(&meta));
    }
}
