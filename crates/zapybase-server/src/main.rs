use axum::{
    extract::{State, Json, Path, Query},
    routing::{get, post, delete},
    Router,
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;
use zapybase_core::{Config, Database, DistanceMetric, QuantizationType};

#[derive(Clone)]
struct AppState {
    db: Arc<Database>,
    start_time: Instant,
}

#[derive(Deserialize)]
struct CreateCollectionRequest {
    name: String,
    dimensions: usize,
    #[serde(default)]
    distance_metric: DistanceMetric,
    #[serde(default)]
    quantization: Option<QuantizationType>,
}

#[derive(Deserialize)]
struct InsertRequest {
    id: String,
    vector: Vec<f32>,
    metadata: Option<Value>,
}

#[derive(Deserialize)]
struct BatchInsertRequest {
    vectors: Vec<InsertRequest>,
}

#[derive(Deserialize)]
struct SearchRequest {
    vector: Vec<f32>,
    k: usize,
}

#[derive(Serialize)]
struct SearchResult {
    id: String,
    distance: f32,
    metadata: Option<Value>,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Serialize)]
struct StatsResponse {
    uptime_seconds: u64,
    database: zapybase_core::DatabaseStats,
}

#[derive(Deserialize)]
struct PaginationParams {
    offset: Option<usize>,
    limit: Option<usize>,
}

#[derive(Serialize)]
struct VectorResponse {
    id: String,
    vector: Vec<f32>,
    metadata: Option<Value>,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let db = Database::new();
    let state = AppState {
        db: Arc::new(db),
        start_time: Instant::now(),
    };

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/stats", get(get_stats))
        .route("/collections", post(create_collection).get(list_collections))
        .route("/collections/:name", delete(delete_collection))
        .route("/collections/:name/vectors", post(insert_vector).get(list_vectors))
        .route("/collections/:name/vectors/batch", post(batch_insert_vector))
        .route("/collections/:name/upsert", post(upsert_vector))
        .route("/collections/:name/vectors/:id", get(get_vector))
        .route("/collections/:name/search", post(search_vector))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    println!("Server listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn health_check() -> &'static str {
    "OK"
}

async fn get_stats(State(state): State<AppState>) -> Json<StatsResponse> {
    let stats = state.db.get_stats();
    let uptime = state.start_time.elapsed().as_secs();
    Json(StatsResponse {
        uptime_seconds: uptime,
        database: stats,
    })
}

async fn create_collection(
    State(state): State<AppState>,
    Json(payload): Json<CreateCollectionRequest>,
) -> Result<&'static str, (StatusCode, Json<ErrorResponse>)> {
    let config = Config {
        dimensions: payload.dimensions,
        distance_metric: payload.distance_metric,
        quantization: payload.quantization.unwrap_or(QuantizationType::None),
        ..Config::default()
    };

    match state.db.create_collection(&payload.name, config) {
        Ok(_) => Ok("Created"),
        Err(e) => Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse { error: e.to_string() }),
        )),
    }
}

async fn delete_collection(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<&'static str, (StatusCode, Json<ErrorResponse>)> {
    match state.db.delete_collection(&name) {
        Ok(_) => Ok("Deleted"),
        Err(e) => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse { error: e.to_string() }),
        )),
    }
}

async fn list_collections(State(state): State<AppState>) -> Json<Vec<String>> {
    Json(state.db.list_collections())
}

async fn insert_vector(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(payload): Json<InsertRequest>,
) -> Result<&'static str, (StatusCode, Json<ErrorResponse>)> {
    let collection = state.db.get_collection(&name).map_err(|e| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse { error: e.to_string() }),
        )
    })?;

    let result = tokio::task::spawn_blocking(move || {
        collection.insert(payload.id, &payload.vector, payload.metadata)
    }).await.map_err(|e| (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse { error: e.to_string() }),
    ))?;

    match result {
        Ok(_) => Ok("Inserted"),
        Err(e) => Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse { error: e.to_string() }),
        )),
    }
}

async fn upsert_vector(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(payload): Json<InsertRequest>,
) -> Result<&'static str, (StatusCode, Json<ErrorResponse>)> {
    let collection = state.db.get_collection(&name).map_err(|e| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse { error: e.to_string() }),
        )
    })?;

    let result = tokio::task::spawn_blocking(move || {
        collection.upsert(payload.id, &payload.vector, payload.metadata)
    }).await.map_err(|e| (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse { error: e.to_string() }),
    ))?;

    match result {
        Ok(_) => Ok("Upserted"),
        Err(e) => Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse { error: e.to_string() }),
        )),
    }
}

async fn batch_insert_vector(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(payload): Json<BatchInsertRequest>,
) -> Result<Json<usize>, (StatusCode, Json<ErrorResponse>)> {
    let collection = state.db.get_collection(&name).map_err(|e| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse { error: e.to_string() }),
        )
    })?;

    let count = payload.vectors.len();
    let result = tokio::task::spawn_blocking(move || {
        for item in payload.vectors {
            // Use upsert for batch to be safe
            collection.upsert(item.id, &item.vector, item.metadata)?;
        }
        Ok::<(), zapybase_core::Error>(())
    }).await.map_err(|e| (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse { error: e.to_string() }),
    ))?;

    match result {
        Ok(_) => Ok(Json(count)),
        Err(e) => Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse { error: e.to_string() }),
        )),
    }
}

async fn get_vector(
    State(state): State<AppState>,
    Path((name, id)): Path<(String, String)>,
) -> Result<Json<VectorResponse>, (StatusCode, Json<ErrorResponse>)> {
    let collection = state.db.get_collection(&name).map_err(|e| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse { error: e.to_string() }),
        )
    })?;

    let id_clone = id.clone();
    let result = tokio::task::spawn_blocking(move || {
        collection.get(&id_clone)
    }).await.map_err(|e| (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse { error: e.to_string() }),
    ))?;

    match result {
        Ok(Some((vector, metadata))) => Ok(Json(VectorResponse {
            id,
            vector,
            metadata,
        })),
        Ok(None) => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse { error: "Vector not found".to_string() }),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse { error: e.to_string() }),
        )),
    }
}

async fn list_vectors(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Query(params): Query<PaginationParams>,
) -> Result<Json<Vec<String>>, (StatusCode, Json<ErrorResponse>)> {
    let collection = state.db.get_collection(&name).map_err(|e| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse { error: e.to_string() }),
        )
    })?;

    let offset = params.offset.unwrap_or(0);
    let limit = params.limit.unwrap_or(10).min(100); // Max 100

    let result = tokio::task::spawn_blocking(move || {
        collection.list(offset, limit)
    }).await.map_err(|e| (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse { error: e.to_string() }),
    ))?;

    Ok(Json(result.into_iter().map(|id| id.to_string()).collect()))
}

async fn search_vector(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(payload): Json<SearchRequest>,
) -> Result<Json<Vec<SearchResult>>, (StatusCode, Json<ErrorResponse>)> {
    let collection = state.db.get_collection(&name).map_err(|e| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse { error: e.to_string() }),
        )
    })?;

    let result = tokio::task::spawn_blocking(move || {
        collection.search(&payload.vector, payload.k)
    }).await.map_err(|e| (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse { error: e.to_string() }),
    ))?;

    match result {
        Ok(results) => {
            let response = results
                .into_iter()
                .map(|(id, distance, metadata)| SearchResult {
                    id: id.as_str().to_string(),
                    distance,
                    metadata,
                })
                .collect();
            Ok(Json(response))
        }
        Err(e) => Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse { error: e.to_string() }),
        )),
    }
}
