//! System-specific type definitions

use std::collections::HashMap;
use std::time::Duration;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
use crate::HealthStatus;

/// System health status with component details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealthStatus {
    pub system_id: Uuid,
    pub status: HealthStatus,
    pub components: HashMap<String, ComponentHealth>,
    pub uptime: Duration,
    pub timestamp: DateTime<Utc>,
}

/// Individual component health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub name: String,
    pub status: HealthStatus,
    pub latency_ms: u64,
    pub last_check: DateTime<Utc>,
}

/// System metrics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub start_time: Option<DateTime<Utc>>,
    pub queries_processed: u64,
    pub queries_successful: u64,
    pub queries_failed: u64,
    pub component_metrics: HashMap<String, ComponentMetrics>,
}

impl SystemMetrics {
    pub fn success_rate(&self) -> f64 {
        if self.queries_processed == 0 {
            0.0
        } else {
            self.queries_successful as f64 / self.queries_processed as f64
        }
    }
}

/// Component-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentMetrics {
    pub requests: u64,
    pub successes: u64,
    pub failures: u64,
    pub avg_latency_ms: f64,
    pub circuit_breaker_state: String,
}

/// Message priorities for the message bus
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MessagePriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Delivery guarantees for messages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeliveryGuarantee {
    AtMostOnce,
    AtLeastOnce,
    ExactlyOnce,
}

/// Generic message wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: Uuid,
    pub priority: MessagePriority,
    pub guarantee: DeliveryGuarantee,
    pub payload: Vec<u8>,
    pub timestamp: DateTime<Utc>,
}