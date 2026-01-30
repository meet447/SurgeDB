use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use std::cmp::Ordering;

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
    /// Range filter for numeric values
    Range {
        field: String,
        gt: Option<f64>,
        gte: Option<f64>,
        lt: Option<f64>,
        lte: Option<f64>,
    },
    /// Geo-spatial radius filter (Haversine distance)
    GeoRadius {
        field: String,
        center: (f64, f64),
        radius_meters: f64,
    },
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
            Filter::Range {
                field,
                gt,
                gte,
                lt,
                lte,
            } => {
                if let Some(value) = get_value_by_path(metadata, field) {
                    if let Some(num) = value.as_f64() {
                        if let Some(limit) = gt {
                            if num.partial_cmp(limit) != Some(Ordering::Greater) {
                                return false;
                            }
                        }
                        if let Some(limit) = gte {
                            if !matches!(
                                num.partial_cmp(limit),
                                Some(Ordering::Greater | Ordering::Equal)
                            ) {
                                return false;
                            }
                        }
                        if let Some(limit) = lt {
                            if num.partial_cmp(limit) != Some(Ordering::Less) {
                                return false;
                            }
                        }
                        if let Some(limit) = lte {
                            if !matches!(
                                num.partial_cmp(limit),
                                Some(Ordering::Less | Ordering::Equal)
                            ) {
                                return false;
                            }
                        }
                        true
                    } else {
                        false // Not a number
                    }
                } else {
                    false // Field missing
                }
            }
            Filter::GeoRadius {
                field,
                center,
                radius_meters,
            } => {
                if let Some(value) = get_value_by_path(metadata, field) {
                    // Expecting { "lat": 1.0, "lon": 2.0 } or [lat, lon]
                    let point = parse_geo_point(value);
                    if let Some((lat, lon)) = point {
                        let dist = haversine_distance(center.0, center.1, lat, lon);
                        dist <= *radius_meters
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
        }
    }
}

fn parse_geo_point(value: &Value) -> Option<(f64, f64)> {
    match value {
        Value::Object(map) => {
            let lat = map.get("lat").or_else(|| map.get("latitude"))?.as_f64()?;
            let lon = map.get("lon").or_else(|| map.get("longitude"))?.as_f64()?;
            Some((lat, lon))
        }
        Value::Array(arr) if arr.len() == 2 => {
            let lat = arr[0].as_f64()?;
            let lon = arr[1].as_f64()?;
            Some((lat, lon))
        }
        _ => None,
    }
}

fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const R: f64 = 6371000.0; // Earth radius in meters
    let d_lat = (lat2 - lat1).to_radians();
    let d_lon = (lon2 - lon1).to_radians();
    let a = (d_lat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (d_lon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
    R * c
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
