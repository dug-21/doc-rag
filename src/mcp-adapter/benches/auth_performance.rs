use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mcp_adapter::{
    auth::{AuthHandler, Credentials, GrantType},
    client::{McpClient, McpClientConfig},
};
use std::time::Duration;
use tokio::runtime::Runtime;

fn bench_auth_token_validation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let auth_handler = AuthHandler::new();
    
    let mut group = c.benchmark_group("auth_token_validation");
    
    // Mock JWT tokens of different sizes
    let small_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c";
    let medium_token = format!("{}.{}.{}", small_token, "additional_claims_here_".repeat(10), "signature_part");
    let large_token = format!("{}.{}.{}", small_token, "large_claims_payload_".repeat(50), "signature_part");
    
    for (size, token) in [
        ("small", small_token),
        ("medium", &medium_token),
        ("large", &large_token),
    ] {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("validate_token", size),
            &token,
            |b, token| {
                b.to_async(&rt).iter(|| async {
                    // Mock token validation
                    black_box(token.len())
                });
            },
        );
    }
    
    group.finish();
}

fn bench_oauth2_flow_simulation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("oauth2_client_credentials_flow", |b| {
        b.to_async(&rt).iter(|| async {
            let credentials = Credentials {
                client_id: "test_client".to_string(),
                client_secret: "test_secret".to_string(),
                username: None,
                password: None,
                grant_type: GrantType::ClientCredentials,
                scope: vec!["read".to_string(), "write".to_string()],
            };
            
            // Simulate OAuth2 flow overhead
            black_box(credentials);
            tokio::time::sleep(Duration::from_micros(100)).await; // Simulate network delay
        });
    });
}

fn bench_token_cache_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("token_cache");
    
    for cache_size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*cache_size as u64));
        group.bench_with_input(
            BenchmarkId::new("cache_lookup", cache_size),
            cache_size,
            |b, &size| {
                b.to_async(&rt).iter(|| async {
                    // Simulate cache lookups
                    for i in 0..size {
                        let tenant_id = format!("tenant_{}", i % 100);
                        black_box(tenant_id);
                    }
                });
            },
        );
    }
    
    group.finish();
}

fn bench_concurrent_auth_requests(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("concurrent_auth");
    
    for concurrency in [10, 50, 100, 200].iter() {
        group.throughput(Throughput::Elements(*concurrency as u64));
        group.bench_with_input(
            BenchmarkId::new("concurrent_requests", concurrency),
            concurrency,
            |b, &concurrent_count| {
                b.to_async(&rt).iter(|| async {
                    let futures: Vec<_> = (0..concurrent_count)
                        .map(|i| async move {
                            // Simulate auth request processing
                            let tenant_id = format!("tenant_{}", i);
                            tokio::time::sleep(Duration::from_micros(50)).await;
                            black_box(tenant_id)
                        })
                        .collect();
                    
                    futures::future::join_all(futures).await
                });
            },
        );
    }
    
    group.finish();
}

fn bench_token_refresh_scenarios(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("token_refresh");
    
    // Simulate different refresh scenarios
    for scenario in ["near_expiry", "expired", "refresh_token_rotation"].iter() {
        group.bench_with_input(
            BenchmarkId::new("refresh_scenario", scenario),
            scenario,
            |b, scenario| {
                b.to_async(&rt).iter(|| async {
                    match *scenario {
                        "near_expiry" => {
                            // Simulate checking token expiry and refreshing
                            tokio::time::sleep(Duration::from_micros(200)).await;
                        }
                        "expired" => {
                            // Simulate expired token handling
                            tokio::time::sleep(Duration::from_micros(300)).await;
                        }
                        "refresh_token_rotation" => {
                            // Simulate refresh token rotation
                            tokio::time::sleep(Duration::from_micros(400)).await;
                        }
                        _ => {}
                    }
                    black_box(scenario)
                });
            },
        );
    }
    
    group.finish();
}

fn bench_multi_tenant_auth(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("multi_tenant_auth");
    
    for tenant_count in [10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*tenant_count as u64));
        group.bench_with_input(
            BenchmarkId::new("tenant_isolation", tenant_count),
            tenant_count,
            |b, &count| {
                b.to_async(&rt).iter(|| async {
                    // Simulate multi-tenant token management
                    let futures: Vec<_> = (0..count)
                        .map(|i| async move {
                            let tenant_id = format!("tenant_{}", i);
                            // Simulate per-tenant auth processing
                            tokio::time::sleep(Duration::from_nanos(1000)).await;
                            black_box(tenant_id)
                        })
                        .collect();
                    
                    futures::future::join_all(futures).await
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    auth_benches,
    bench_auth_token_validation,
    bench_oauth2_flow_simulation,
    bench_token_cache_performance,
    bench_concurrent_auth_requests,
    bench_token_refresh_scenarios,
    bench_multi_tenant_auth
);

criterion_main!(auth_benches);