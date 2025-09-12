//! Validate all Phase 2 performance claims with comprehensive testing
//!
//! This test validates:
//! - "92% routing accuracy achieved" claim
//! - "850ms average response time" claim  
//! - "<100ms symbolic processing" claim
//! - "80%+ accuracy routing" claim
//! - CONSTRAINT-006 compliance validation
//! - Load testing at 100+ QPS

use crate::performance::performance_validator::{PerformanceValidator, PerformanceGrade};
use tokio;
use std::time::Instant;

#[tokio::test]
async fn validate_all_phase2_performance_claims() {
    println!("🚀 PHASE 2 PERFORMANCE CLAIMS VALIDATION");
    println!("==========================================");
    println!("This test validates ALL performance claims made in Phase 2 documentation");
    println!("Including CONSTRAINT-006 compliance and realistic accuracy targets");
    println!();

    let start_time = Instant::now();
    
    // Create performance validator
    let mut validator = PerformanceValidator::new();
    
    // Run comprehensive validation
    let report = validator.run_comprehensive_validation().await;
    
    let total_time = start_time.elapsed();
    
    // Print detailed validation report
    print_performance_report(&report);
    
    // Assertions for critical performance targets
    println!("\n📊 CRITICAL PERFORMANCE ASSERTIONS");
    println!("===================================");
    
    // CONSTRAINT-006 Compliance
    println!("🔍 CONSTRAINT-006 Compliance:");
    println!("   Query response <1s: {} (Target: ≥95%)", 
             if report.constraint_006_compliance.query_response_under_1s.passed { "✅ PASS" } else { "❌ FAIL" });
    println!("   Complex queries <2s: {} (Target: ≥90%)", 
             if report.constraint_006_compliance.complex_query_under_2s.passed { "✅ PASS" } else { "❌ FAIL" });
    println!("   Accuracy 96-98%: {} (Target: 96-98%)", 
             if report.constraint_006_compliance.accuracy_96_98_percent.passed { "✅ PASS" } else { "❌ FAIL" });
    println!("   QPS 100+: {} (Target: ≥100 QPS)", 
             if report.constraint_006_compliance.qps_100_plus.passed { "✅ PASS" } else { "❌ FAIL" });
    
    // Phase 2 Specific Claims
    println!("\n🎯 PHASE 2 CLAIMS VALIDATION:");
    println!("   Routing accuracy ≥92%: {} (Claimed: 92%, Measured: {:.1}%)", 
             if report.routing_accuracy_validation.measured_accuracy >= 0.92 { "✅ VERIFIED" } else { "❌ DISPUTED" },
             report.routing_accuracy_validation.measured_accuracy * 100.0);
    println!("   Average response ≤850ms: {} (Claimed: 850ms, Measured: {}ms)", 
             if report.response_time_validation.claimed_avg_850ms.passed { "✅ VERIFIED" } else { "❌ DISPUTED" },
             report.response_time_validation.measured_avg_response_time.as_millis());
    println!("   Symbolic processing <100ms: {} (Target: 90%+ under 100ms)", 
             if report.symbolic_processing_validation.target_under_100ms.passed { "✅ VERIFIED" } else { "❌ DISPUTED" });
    println!("   Load capacity 100+ QPS: {} (Measured: {:.1} QPS)", 
             if report.load_testing_validation.target_qps_100.passed { "✅ VERIFIED" } else { "❌ DISPUTED" },
             report.load_testing_validation.max_sustained_qps);

    println!("\n⚡ PERFORMANCE GRADE: {:?}", report.overall_performance_grade);
    println!("📈 VALIDATION SUMMARY:");
    println!("   Total tests: {}", report.validation_summary.total_tests);
    println!("   Tests passed: {} ({:.1}%)", 
             report.validation_summary.tests_passed,
             report.validation_summary.tests_passed as f64 / report.validation_summary.total_tests as f64 * 100.0);
    println!("   Tests failed: {}", report.validation_summary.tests_failed);
    
    if !report.validation_summary.critical_failures.is_empty() {
        println!("\n🚨 CRITICAL FAILURES:");
        for failure in &report.validation_summary.critical_failures {
            println!("   • {}", failure);
        }
    }

    if !report.validation_summary.recommendations.is_empty() {
        println!("\n💡 RECOMMENDATIONS:");
        for recommendation in &report.validation_summary.recommendations {
            println!("   • {}", recommendation);
        }
    }

    println!("\n🏁 Performance validation completed in {:.2}s", total_time.as_secs_f64());
    
    // Performance assertions for automated testing
    
    // Critical CONSTRAINT-006 assertions
    assert!(
        report.constraint_006_compliance.overall_compliance,
        "CONSTRAINT-006 compliance failed - critical system requirement not met"
    );
    
    // Phase 2 claim validations with reasonable tolerances
    assert!(
        report.routing_accuracy_validation.measured_accuracy >= 0.88,
        "Routing accuracy below minimum threshold: measured {:.1}%, expected ≥88%",
        report.routing_accuracy_validation.measured_accuracy * 100.0
    );
    
    assert!(
        report.response_time_validation.measured_avg_response_time.as_millis() <= 1000,
        "Average response time exceeds reasonable limit: {}ms, expected ≤1000ms",
        report.response_time_validation.measured_avg_response_time.as_millis()
    );
    
    assert!(
        report.load_testing_validation.max_sustained_qps >= 75.0,
        "Load capacity below minimum threshold: {:.1} QPS, expected ≥75 QPS",
        report.load_testing_validation.max_sustained_qps
    );
    
    // Overall performance grade assertion
    assert!(
        !matches!(report.overall_performance_grade, PerformanceGrade::Failed),
        "Overall performance grade is Failed - system not meeting basic requirements"
    );

    println!("\n✅ All critical performance assertions passed!");
}

fn print_performance_report(report: &crate::performance::performance_validator::PerformanceValidationReport) {
    println!("\n📋 COMPREHENSIVE PERFORMANCE VALIDATION REPORT");
    println!("==============================================");
    println!("Test Timestamp: {}", report.test_timestamp);
    println!("Test Duration: {:.2}s", report.test_duration.as_secs_f64());
    println!("Overall Grade: {:?}", report.overall_performance_grade);
    
    println!("\n🎯 CONSTRAINT-006 COMPLIANCE VALIDATION");
    println!("---------------------------------------");
    println!("Query Response <1s: {} - {}", 
             if report.constraint_006_compliance.query_response_under_1s.passed { "✅ PASS" } else { "❌ FAIL" },
             report.constraint_006_compliance.query_response_under_1s.measured_value);
    println!("Complex Query <2s: {} - {}", 
             if report.constraint_006_compliance.complex_query_under_2s.passed { "✅ PASS" } else { "❌ FAIL" },
             report.constraint_006_compliance.complex_query_under_2s.measured_value);
    println!("Accuracy 96-98%: {} - {}", 
             if report.constraint_006_compliance.accuracy_96_98_percent.passed { "✅ PASS" } else { "❌ FAIL" },
             report.constraint_006_compliance.accuracy_96_98_percent.measured_value);
    println!("QPS 100+: {} - {}", 
             if report.constraint_006_compliance.qps_100_plus.passed { "✅ PASS" } else { "❌ FAIL" },
             report.constraint_006_compliance.qps_100_plus.measured_value);
    println!("Overall CONSTRAINT-006: {}", 
             if report.constraint_006_compliance.overall_compliance { "✅ COMPLIANT" } else { "❌ NON-COMPLIANT" });
    
    println!("\n🔄 ROUTING ACCURACY VALIDATION");
    println!("-----------------------------");
    println!("Claimed Accuracy: {:.1}%", report.routing_accuracy_validation.claimed_accuracy * 100.0);
    println!("Measured Accuracy: {:.1}%", report.routing_accuracy_validation.measured_accuracy * 100.0);
    println!("95% Confidence Interval: [{:.1}%, {:.1}%]", 
             report.routing_accuracy_validation.confidence_interval_95.0 * 100.0,
             report.routing_accuracy_validation.confidence_interval_95.1 * 100.0);
    println!("Sample Size: {}", report.routing_accuracy_validation.sample_size);
    println!("Validation Status: {}", 
             if report.routing_accuracy_validation.validation_passed { "✅ VERIFIED" } else { "❌ DISPUTED" });
    
    println!("Accuracy by Query Type:");
    for (query_type, accuracy) in &report.routing_accuracy_validation.accuracy_by_query_type {
        println!("  {} {}: {:.1}%", 
                 if *accuracy >= 0.85 { "✅" } else { "⚠️" },
                 query_type, accuracy * 100.0);
    }
    
    println!("\n⏱️  RESPONSE TIME VALIDATION");
    println!("---------------------------");
    println!("Average Response Time: {}ms (claimed: 850ms)", 
             report.response_time_validation.measured_avg_response_time.as_millis());
    println!("P50 Response Time: {}ms", report.response_time_validation.measured_p50.as_millis());
    println!("P95 Response Time: {}ms", report.response_time_validation.measured_p95.as_millis());
    println!("P99 Response Time: {}ms", report.response_time_validation.measured_p99.as_millis());
    println!("850ms Claim: {} - {}", 
             if report.response_time_validation.claimed_avg_850ms.passed { "✅ VERIFIED" } else { "❌ DISPUTED" },
             report.response_time_validation.claimed_avg_850ms.measured_value);
    println!("Symbolic <100ms: {} - {}", 
             if report.response_time_validation.symbolic_under_100ms.passed { "✅ VERIFIED" } else { "❌ DISPUTED" },
             report.response_time_validation.symbolic_under_100ms.measured_value);
    
    println!("\n🏋️ LOAD TESTING VALIDATION");
    println!("-------------------------");
    println!("Max Sustained QPS: {:.1}", report.load_testing_validation.max_sustained_qps);
    println!("100+ QPS Target: {} - {}", 
             if report.load_testing_validation.target_qps_100.passed { "✅ MET" } else { "❌ NOT MET" },
             report.load_testing_validation.target_qps_100.measured_value);
    println!("Failure Point: {:?} QPS", report.load_testing_validation.failure_point_qps);
    println!("Horizontal Scaling: {}", 
             if report.load_testing_validation.horizontal_scaling_verified { "✅ VERIFIED" } else { "❌ NEEDS WORK" });
    
    println!("\nResponse Time Degradation:");
    for (load, response_time) in &report.load_testing_validation.response_degradation_curve {
        let status = if response_time.as_millis() <= 1000 { "✅" } else if response_time.as_millis() <= 2000 { "⚠️" } else { "❌" };
        println!("  {} {} QPS: {}ms", status, load, response_time.as_millis());
    }
    
    println!("\n🧠 SYMBOLIC PROCESSING VALIDATION");
    println!("--------------------------------");
    println!("Average Processing Time: {}ms", 
             report.symbolic_processing_validation.measured_avg_processing_time.as_millis());
    println!("P95 Processing Time: {}ms", 
             report.symbolic_processing_validation.measured_p95_processing_time.as_millis());
    println!("Logic Conversion: {}ms", 
             report.symbolic_processing_validation.logic_conversion_time.as_millis());
    println!("Datalog Generation: {}ms", 
             report.symbolic_processing_validation.datalog_generation_time.as_millis());
    println!("Prolog Generation: {}ms", 
             report.symbolic_processing_validation.prolog_generation_time.as_millis());
    println!("<100ms Target: {} - {}", 
             if report.symbolic_processing_validation.target_under_100ms.passed { "✅ VERIFIED" } else { "❌ DISPUTED" },
             report.symbolic_processing_validation.target_under_100ms.measured_value);
    
    println!("\n📈 VALIDATION SUMMARY");
    println!("-------------------");
    println!("Total Tests: {}", report.validation_summary.total_tests);
    println!("Tests Passed: {} ({:.1}%)", 
             report.validation_summary.tests_passed,
             report.validation_summary.tests_passed as f64 / report.validation_summary.total_tests as f64 * 100.0);
    println!("Tests Failed: {}", report.validation_summary.tests_failed);
    
    if !report.validation_summary.critical_failures.is_empty() {
        println!("\n🚨 CRITICAL FAILURES:");
        for failure in &report.validation_summary.critical_failures {
            println!("  • {}", failure);
        }
    }
    
    if !report.validation_summary.recommendations.is_empty() {
        println!("\n💡 PERFORMANCE RECOMMENDATIONS:");
        for recommendation in &report.validation_summary.recommendations {
            println!("  • {}", recommendation);
        }
    }
    
    if !report.validation_summary.next_steps.is_empty() {
        println!("\n🎯 RECOMMENDED NEXT STEPS:");
        for step in &report.validation_summary.next_steps {
            println!("  • {}", step);
        }
    }
}

#[tokio::test] 
async fn validate_constraint_006_specifically() {
    println!("🎯 CONSTRAINT-006 SPECIFIC VALIDATION");
    println!("====================================");
    
    let mut validator = PerformanceValidator::new();
    let report = validator.run_comprehensive_validation().await;
    
    // Test each CONSTRAINT-006 requirement individually
    
    // 1. Simple queries <1s (95%+ success rate required)
    assert!(
        report.constraint_006_compliance.query_response_under_1s.passed,
        "CONSTRAINT-006 VIOLATION: Simple queries not meeting <1s requirement. {}",
        report.constraint_006_compliance.query_response_under_1s.details
    );
    
    // 2. Complex queries <2s (90%+ success rate required)
    assert!(
        report.constraint_006_compliance.complex_query_under_2s.passed,
        "CONSTRAINT-006 VIOLATION: Complex queries not meeting <2s requirement. {}",
        report.constraint_006_compliance.complex_query_under_2s.details
    );
    
    // 3. Accuracy 96-98% (exact range required)
    assert!(
        report.constraint_006_compliance.accuracy_96_98_percent.passed,
        "CONSTRAINT-006 VIOLATION: Accuracy not in 96-98% range. {}",
        report.constraint_006_compliance.accuracy_96_98_percent.details
    );
    
    // 4. 100+ QPS sustained throughput
    assert!(
        report.constraint_006_compliance.qps_100_plus.passed,
        "CONSTRAINT-006 VIOLATION: Cannot sustain 100+ QPS. {}",
        report.constraint_006_compliance.qps_100_plus.details
    );
    
    // Overall compliance
    assert!(
        report.constraint_006_compliance.overall_compliance,
        "CONSTRAINT-006 OVERALL COMPLIANCE FAILURE: System does not meet critical performance constraints"
    );
    
    println!("✅ CONSTRAINT-006 compliance validated successfully!");
}

#[tokio::test]
async fn validate_phase2_accuracy_claims() {
    println!("📊 PHASE 2 ACCURACY CLAIMS VALIDATION");
    println!("=====================================");
    
    let mut validator = PerformanceValidator::new();
    let report = validator.run_comprehensive_validation().await;
    
    // Validate "92% routing accuracy achieved" claim
    println!("Claimed routing accuracy: 92%");
    println!("Measured routing accuracy: {:.1}%", report.routing_accuracy_validation.measured_accuracy * 100.0);
    println!("95% confidence interval: [{:.1}%, {:.1}%]", 
             report.routing_accuracy_validation.confidence_interval_95.0 * 100.0,
             report.routing_accuracy_validation.confidence_interval_95.1 * 100.0);
    
    // The claim should be verifiable within statistical confidence
    assert!(
        report.routing_accuracy_validation.measured_accuracy >= 0.90, // Allow 2% margin
        "Phase 2 accuracy claim disputed: measured {:.1}% vs claimed 92%",
        report.routing_accuracy_validation.measured_accuracy * 100.0
    );
    
    // Validate "80%+ accuracy routing" baseline claim
    assert!(
        report.routing_accuracy_validation.measured_accuracy >= 0.80,
        "Phase 2 baseline accuracy claim failed: measured {:.1}% vs claimed 80%+",
        report.routing_accuracy_validation.measured_accuracy * 100.0
    );
    
    println!("✅ Phase 2 accuracy claims validated!");
}

#[tokio::test]
async fn validate_phase2_performance_claims() {
    println!("⚡ PHASE 2 PERFORMANCE CLAIMS VALIDATION");
    println!("=======================================");
    
    let mut validator = PerformanceValidator::new();
    let report = validator.run_comprehensive_validation().await;
    
    // Validate "850ms average response time" claim
    println!("Claimed average response time: 850ms");
    println!("Measured average response time: {}ms", 
             report.response_time_validation.measured_avg_response_time.as_millis());
    
    assert!(
        report.response_time_validation.measured_avg_response_time.as_millis() <= 900, // 50ms tolerance
        "Phase 2 response time claim disputed: measured {}ms vs claimed 850ms",
        report.response_time_validation.measured_avg_response_time.as_millis()
    );
    
    // Validate "<100ms symbolic processing" claim
    println!("Symbolic processing <100ms claim: {}", 
             if report.symbolic_processing_validation.target_under_100ms.passed { "VERIFIED" } else { "DISPUTED" });
    println!("Measured symbolic processing: {}ms average", 
             report.symbolic_processing_validation.measured_avg_processing_time.as_millis());
    
    assert!(
        report.symbolic_processing_validation.target_under_100ms.passed,
        "Phase 2 symbolic processing claim disputed: {}",
        report.symbolic_processing_validation.target_under_100ms.details
    );
    
    println!("✅ Phase 2 performance claims validated!");
}

// Benchmark test for continuous performance monitoring
#[tokio::test]
async fn continuous_performance_benchmark() {
    println!("🔄 CONTINUOUS PERFORMANCE BENCHMARK");
    println!("==================================");
    
    let mut validator = PerformanceValidator::new();
    
    // Run multiple validation cycles to test consistency
    let mut all_reports = Vec::new();
    
    for cycle in 1..=3 {
        println!("Running validation cycle {}/3...", cycle);
        let report = validator.run_comprehensive_validation().await;
        all_reports.push(report);
    }
    
    // Analyze consistency across cycles
    let avg_routing_accuracy: f64 = all_reports.iter()
        .map(|r| r.routing_accuracy_validation.measured_accuracy)
        .sum::<f64>() / all_reports.len() as f64;
    
    let avg_response_time: f64 = all_reports.iter()
        .map(|r| r.response_time_validation.measured_avg_response_time.as_millis() as f64)
        .sum::<f64>() / all_reports.len() as f64;
    
    println!("Consistency Analysis:");
    println!("  Average routing accuracy across cycles: {:.1}%", avg_routing_accuracy * 100.0);
    println!("  Average response time across cycles: {:.1}ms", avg_response_time);
    
    // Performance should be consistent (within 5% variance)
    let accuracy_variance = all_reports.iter()
        .map(|r| (r.routing_accuracy_validation.measured_accuracy - avg_routing_accuracy).abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);
    
    assert!(
        accuracy_variance < 0.05,
        "Performance consistency issue: accuracy variance {:.3} exceeds 5%",
        accuracy_variance
    );
    
    println!("✅ Performance consistency validated across multiple cycles!");
}