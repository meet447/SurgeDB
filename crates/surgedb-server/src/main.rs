use axum::{
    extract::{Json, Path, Query, Request, State},
    http::{header::HeaderName, HeaderValue, Method, StatusCode},
    middleware::{self, Next},
    response::IntoResponse,
    routing::{delete, get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use surgedb_core::filter::Filter;
use surgedb_core::{Config as DbConfig, Database, DistanceMetric, QuantizationType};
use sysinfo::System;
use tower_http::{
    compression::CompressionLayer, cors::CorsLayer, limit::RequestBodyLimitLayer,
    timeout::TimeoutLayer, trace::TraceLayer,
};
use tracing::{info, warn};
use tracing_subscriber::{fmt, EnvFilter};
use utoipa::{IntoParams, OpenApi, ToSchema};
use utoipa_swagger_ui::SwaggerUi;

// =============================================================================
// Configuration
// =============================================================================

#[derive(Clone)]
struct AppConfig {
    port: u16,
    api_key: Option<String>,
    log_level: String,
    cors_allow_origin: String,
    request_timeout_secs: u64,
    max_request_size_bytes: usize,
}

impl AppConfig {
    fn from_env() -> Self {
        dotenvy::dotenv().ok();
        Self {
            port: std::env::var("PORT")
                .unwrap_or_else(|_| "3000".to_string())
                .parse()
                .unwrap_or(3000),
            api_key: std::env::var("API_KEY").ok(),
            log_level: std::env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
            cors_allow_origin: std::env::var("CORS_ALLOW_ORIGIN")
                .unwrap_or_else(|_| "*".to_string()),
            request_timeout_secs: std::env::var("REQUEST_TIMEOUT_SECS")
                .unwrap_or_else(|_| "30".to_string())
                .parse()
                .unwrap_or(30),
            max_request_size_bytes: std::env::var("MAX_REQUEST_SIZE_BYTES")
                .unwrap_or_else(|_| "10485760".to_string()) // 10MB
                .parse()
                .unwrap_or(10 * 1024 * 1024),
        }
    }
}

// =============================================================================
// State & Models
// =============================================================================

#[derive(Clone)]
struct AppState {
    db: Arc<Database>,
    config: AppConfig,
    start_time: Instant,
}

#[derive(Deserialize, ToSchema)]
struct CreateCollectionRequest {
    #[schema(example = "my_collection")]
    name: String,
    #[schema(example = 384)]
    dimensions: usize,
    #[serde(default)]
    #[schema(example = "Cosine")]
    distance_metric: DistanceMetric,
    #[serde(default)]
    quantization: Option<QuantizationType>,
}

#[derive(Deserialize, ToSchema)]
struct InsertRequest {
    #[schema(example = "vec1")]
    id: String,
    #[schema(example = "[0.1, 0.2, 0.3]")]
    vector: Vec<f32>,
    metadata: Option<Value>,
}

#[derive(Deserialize, ToSchema)]
struct BatchInsertRequest {
    vectors: Vec<InsertRequest>,
}

#[derive(Deserialize, ToSchema)]
struct SearchRequest {
    #[schema(example = "[0.1, 0.2, 0.3]")]
    vector: Vec<f32>,
    #[schema(example = 10)]
    k: usize,
    filter: Option<Filter>,
}

#[derive(Serialize, ToSchema)]
struct SearchResult {
    id: String,
    distance: f32,
    metadata: Option<Value>,
}

#[derive(Serialize, ToSchema)]
struct ErrorResponse {
    error: String,
}

#[derive(Serialize, ToSchema)]
struct HealthResponse {
    status: String,
    version: String,
    uptime_seconds: u64,
    memory_usage_mb: u64,
}

#[derive(Serialize, ToSchema)]
struct StatsResponse {
    uptime_seconds: u64,
    database: surgedb_core::DatabaseStats,
}

#[derive(Deserialize, IntoParams)]
struct PaginationParams {
    #[param(example = 0)]
    offset: Option<usize>,
    #[param(example = 10)]
    limit: Option<usize>,
}

#[derive(Serialize, ToSchema)]
struct VectorResponse {
    id: String,
    vector: Vec<f32>,
    metadata: Option<Value>,
}

// =============================================================================
// OpenAPI Documentation
// =============================================================================

#[derive(OpenApi)]
#[openapi(
    paths(
        health_check,
        get_stats,
        create_collection,
        list_collections,
        delete_collection,
        insert_vector,
        list_vectors,
        batch_insert_vector,
        upsert_vector,
        get_vector,
        delete_vector,
        search_vector,
    ),
    components(
        schemas(
            CreateCollectionRequest, InsertRequest, BatchInsertRequest,
            SearchRequest, SearchResult, ErrorResponse, HealthResponse,
            StatsResponse, VectorResponse
        )
    ),
    tags(
        (name = "surgedb", description = "SurgeDB Vector Search API")
    )
)]
struct ApiDoc;

// =============================================================================
// Middleware
// =============================================================================

async fn auth_middleware(
    State(state): State<AppState>,
    req: Request,
    next: Next,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    if let Some(expected_key) = &state.config.api_key {
        let auth_header = req.headers().get("x-api-key").and_then(|v| v.to_str().ok());

        if auth_header != Some(expected_key) {
            return Err((
                StatusCode::UNAUTHORIZED,
                Json(ErrorResponse {
                    error: "Invalid or missing API key".to_string(),
                }),
            ));
        }
    }
    Ok(next.run(req).await)
}

// =============================================================================
// Main Entry Point
// =============================================================================

#[tokio::main]
async fn main() {
    let config = AppConfig::from_env();

    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&config.log_level));

    fmt().with_env_filter(env_filter).with_target(false).init();

    info!("Starting SurgeDB Server v{}", env!("CARGO_PKG_VERSION"));

    let db = Database::new();
    let state = AppState {
        db: Arc::new(db),
        config: config.clone(),
        start_time: Instant::now(),
    };

    let cors = CorsLayer::new()
        .allow_origin(config.cors_allow_origin.parse::<HeaderValue>().unwrap())
        .allow_methods([Method::GET, Method::POST, Method::DELETE])
        .allow_headers([
            axum::http::header::CONTENT_TYPE,
            HeaderName::from_static("x-api-key"),
        ]);

    let app = Router::new()
        .route("/health", get(health_check))
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .nest(
            "/",
            Router::new()
                .route("/stats", get(get_stats))
                .route(
                    "/collections",
                    post(create_collection).get(list_collections),
                )
                .route("/collections/:name", delete(delete_collection))
                .route(
                    "/collections/:name/vectors",
                    post(insert_vector).get(list_vectors),
                )
                .route(
                    "/collections/:name/vectors/batch",
                    post(batch_insert_vector),
                )
                .route("/collections/:name/upsert", post(upsert_vector))
                .route(
                    "/collections/:name/vectors/:id",
                    get(get_vector).delete(delete_vector),
                )
                .route("/collections/:name/search", post(search_vector))
                .layer(middleware::from_fn_with_state(
                    state.clone(),
                    auth_middleware,
                )),
        )
        .layer(TraceLayer::new_for_http())
        .layer(CompressionLayer::new())
        .layer(TimeoutLayer::new(Duration::from_secs(
            config.request_timeout_secs,
        )))
        .layer(RequestBodyLimitLayer::new(config.max_request_size_bytes))
        .layer(cors)
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], config.port));
    info!("Server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => info!("Received Ctrl+C, shutting down..."),
        _ = terminate => info!("Received SIGTERM, shutting down..."),
    }
}

// =============================================================================
// Route Handlers
// =============================================================================

#[utoipa::path(
    get,
    path = "/health",
    responses(
        (status = 200, description = "Server is healthy", body = HealthResponse)
    )
)]
async fn health_check(State(state): State<AppState>) -> Json<HealthResponse> {
    let mut sys = System::new_all();
    sys.refresh_all();

    // Get current process memory usage instead of total system memory
    let pid = sysinfo::get_current_pid().ok();
    let process_memory = pid
        .and_then(|p| sys.process(p))
        .map(|p| p.memory())
        .unwrap_or(0);

    Json(HealthResponse {
        status: "OK".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: state.start_time.elapsed().as_secs(),
        memory_usage_mb: process_memory / 1024 / 1024,
    })
}

#[utoipa::path(
    get,
    path = "/stats",
    responses(
        (status = 200, description = "Database statistics", body = StatsResponse)
    ),
    security(("api_key" = []))
)]
async fn get_stats(State(state): State<AppState>) -> Json<StatsResponse> {
    let stats = state.db.get_stats();
    let uptime = state.start_time.elapsed().as_secs();
    Json(StatsResponse {
        uptime_seconds: uptime,
        database: stats,
    })
}

#[utoipa::path(
    post,
    path = "/collections",
    request_body = CreateCollectionRequest,
    responses(
        (status = 200, description = "Collection created"),
        (status = 400, description = "Invalid request", body = ErrorResponse)
    ),
    security(("api_key" = []))
)]
async fn create_collection(
    State(state): State<AppState>,
    Json(payload): Json<CreateCollectionRequest>,
) -> Result<&'static str, (StatusCode, Json<ErrorResponse>)> {
    let config = DbConfig {
        dimensions: payload.dimensions,
        distance_metric: payload.distance_metric,
        quantization: payload.quantization.unwrap_or(QuantizationType::None),
        ..DbConfig::default()
    };

    match state.db.create_collection(&payload.name, config) {
        Ok(_) => {
            info!("Created collection: {}", payload.name);
            Ok("Created")
        }
        Err(e) => {
            warn!("Failed to create collection {}: {}", payload.name, e);
            Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            ))
        }
    }
}

#[utoipa::path(
    get,
    path = "/collections",
    responses(
        (status = 200, description = "List of collection names", body = [String])
    ),
    security(("api_key" = []))
)]
async fn list_collections(State(state): State<AppState>) -> Json<Vec<String>> {
    Json(state.db.list_collections())
}

#[utoipa::path(
    delete,
    path = "/collections/{name}",
    params(
        ("name" = String, Path, description = "Collection name")
    ),
    responses(
        (status = 200, description = "Collection deleted"),
        (status = 404, description = "Collection not found", body = ErrorResponse)
    ),
    security(("api_key" = []))
)]
async fn delete_collection(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<&'static str, (StatusCode, Json<ErrorResponse>)> {
    match state.db.delete_collection(&name) {
        Ok(_) => {
            info!("Deleted collection: {}", name);
            Ok("Deleted")
        }
        Err(e) => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )),
    }
}

#[utoipa::path(
    post,
    path = "/collections/{name}/vectors",
    params(
        ("name" = String, Path, description = "Collection name")
    ),
    request_body = InsertRequest,
    responses(
        (status = 200, description = "Vector inserted"),
        (status = 400, description = "Invalid request", body = ErrorResponse),
        (status = 404, description = "Collection not found", body = ErrorResponse)
    ),
    security(("api_key" = []))
)]
async fn insert_vector(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(payload): Json<InsertRequest>,
) -> Result<&'static str, (StatusCode, Json<ErrorResponse>)> {
    let collection = state.db.get_collection(&name).map_err(|e| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    let result = tokio::task::spawn_blocking(move || {
        collection.insert(payload.id, &payload.vector, payload.metadata)
    })
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    match result {
        Ok(_) => Ok("Inserted"),
        Err(e) => Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )),
    }
}

#[utoipa::path(
    post,
    path = "/collections/{name}/upsert",
    params(
        ("name" = String, Path, description = "Collection name")
    ),
    request_body = InsertRequest,
    responses(
        (status = 200, description = "Vector upserted"),
        (status = 400, description = "Invalid request", body = ErrorResponse)
    ),
    security(("api_key" = []))
)]
async fn upsert_vector(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(payload): Json<InsertRequest>,
) -> Result<&'static str, (StatusCode, Json<ErrorResponse>)> {
    let collection = state.db.get_collection(&name).map_err(|e| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    let result = tokio::task::spawn_blocking(move || {
        collection.upsert(payload.id, &payload.vector, payload.metadata)
    })
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    match result {
        Ok(_) => Ok("Upserted"),
        Err(e) => Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )),
    }
}

#[utoipa::path(
    post,
    path = "/collections/{name}/vectors/batch",
    params(
        ("name" = String, Path, description = "Collection name")
    ),
    request_body = BatchInsertRequest,
    responses(
        (status = 200, description = "Number of vectors upserted", body = usize),
        (status = 400, description = "Invalid request", body = ErrorResponse)
    ),
    security(("api_key" = []))
)]
async fn batch_insert_vector(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(payload): Json<BatchInsertRequest>,
) -> Result<Json<usize>, (StatusCode, Json<ErrorResponse>)> {
    let collection = state.db.get_collection(&name).map_err(|e| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    let count = payload.vectors.len();
    let result = tokio::task::spawn_blocking(move || {
        let items: Vec<(String, Vec<f32>, Option<Value>)> = payload
            .vectors
            .into_iter()
            .map(|item| (item.id, item.vector, item.metadata))
            .collect();

        collection.upsert_batch(items)?;
        Ok::<(), surgedb_core::Error>(())
    })
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    match result {
        Ok(_) => Ok(Json(count)),
        Err(e) => Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )),
    }
}

#[utoipa::path(
    get,
    path = "/collections/{name}/vectors/{id}",
    params(
        ("name" = String, Path, description = "Collection name"),
        ("id" = String, Path, description = "Vector ID")
    ),
    responses(
        (status = 200, description = "Vector found", body = VectorResponse),
        (status = 404, description = "Vector not found", body = ErrorResponse)
    ),
    security(("api_key" = []))
)]
async fn get_vector(
    State(state): State<AppState>,
    Path((name, id)): Path<(String, String)>,
) -> Result<Json<VectorResponse>, (StatusCode, Json<ErrorResponse>)> {
    let collection = state.db.get_collection(&name).map_err(|e| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    let id_clone = id.clone();
    let result = tokio::task::spawn_blocking(move || collection.get(&id_clone))
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })?;

    match result {
        Ok(Some((vector, metadata))) => Ok(Json(VectorResponse {
            id,
            vector,
            metadata,
        })),
        Ok(None) => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Vector not found".to_string(),
            }),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )),
    }
}

#[utoipa::path(
    delete,
    path = "/collections/{name}/vectors/{id}",
    params(
        ("name" = String, Path, description = "Collection name"),
        ("id" = String, Path, description = "Vector ID")
    ),
    responses(
        (status = 200, description = "Vector deleted"),
        (status = 404, description = "Vector not found", body = ErrorResponse)
    ),
    security(("api_key" = []))
)]
async fn delete_vector(
    State(state): State<AppState>,
    Path((name, id)): Path<(String, String)>,
) -> Result<&'static str, (StatusCode, Json<ErrorResponse>)> {
    let collection = state.db.get_collection(&name).map_err(|e| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    let id_clone = id.clone();
    let result = tokio::task::spawn_blocking(move || collection.delete(&id_clone))
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })?;

    match result {
        Ok(true) => Ok("Deleted"),
        Ok(false) => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Vector not found".to_string(),
            }),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )),
    }
}

#[utoipa::path(
    get,
    path = "/collections/{name}/vectors",
    params(
        ("name" = String, Path, description = "Collection name"),
        PaginationParams
    ),
    responses(
        (status = 200, description = "List of vector IDs", body = [String])
    ),
    security(("api_key" = []))
)]
async fn list_vectors(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Query(params): Query<PaginationParams>,
) -> Result<Json<Vec<String>>, (StatusCode, Json<ErrorResponse>)> {
    let collection = state.db.get_collection(&name).map_err(|e| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    let offset = params.offset.unwrap_or(0);
    let limit = params.limit.unwrap_or(10).min(100);

    let result = tokio::task::spawn_blocking(move || collection.list(offset, limit))
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })?;

    Ok(Json(result.into_iter().map(|id| id.to_string()).collect()))
}

#[utoipa::path(
    post,
    path = "/collections/{name}/search",
    params(
        ("name" = String, Path, description = "Collection name")
    ),
    request_body = SearchRequest,
    responses(
        (status = 200, description = "List of nearest neighbors", body = [SearchResult]),
        (status = 400, description = "Invalid request", body = ErrorResponse)
    ),
    security(("api_key" = []))
)]
async fn search_vector(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(payload): Json<SearchRequest>,
) -> Result<Json<Vec<SearchResult>>, (StatusCode, Json<ErrorResponse>)> {
    let collection = state.db.get_collection(&name).map_err(|e| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    let result = tokio::task::spawn_blocking(move || {
        collection.search(&payload.vector, payload.k, payload.filter.as_ref())
    })
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

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
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )),
    }
}
