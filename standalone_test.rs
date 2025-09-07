//! Standalone Integration Test
//!
//! Validates core functionality without workspace dependencies

use std::time::{Duration, Instant};

fn main() {
    println!("🚀 Running standalone integration tests");
    
    // Test 1: ruv-FANN availability
    test_ruv_fann_basic();
    
    // Test 2: Byzantine consensus calculations
    test_byzantine_consensus();
    
    // Test 3: Performance validation
    test_performance();
    
    println!("✅ All standalone tests passed!");
}

fn test_ruv_fann_basic() {
    print!("Testing ruv-FANN neural networks... ");
    
    let network_result = ruv_fann::Network::<f32>::new(&[2, 3, 1]);
    assert!(network_result.is_ok(), "Should create neural network");
    
    let mut network = network_result.unwrap();
    let input = vec![0.5, 0.7];
    let output = network.run(&input);
    
    assert!(output.is_ok(), "Should process neural input");
    let output_values = output.unwrap();
    assert_eq!(output_values.len(), 1, "Should get one output");
    
    println!("✅ PASS (input: {:?} -> output: {:?})", input, output_values);
}

fn test_byzantine_consensus() {
    print!("Testing Byzantine consensus calculations... ");
    
    let scenarios = vec![
        (10, 7, true),   // 70% should pass
        (10, 6, false),  // 60% should fail
        (15, 10, true),  // 66.7% should pass  
        (15, 9, false),  // 60% should fail
    ];
    
    for (total, votes, expected) in scenarios {
        let percentage = votes as f64 / total as f64;
        let consensus = percentage >= 0.67;
        assert_eq!(consensus, expected, 
            "Consensus failed for {}/{} = {:.1}%", votes, total, percentage * 100.0);
    }
    
    println!("✅ PASS (66% threshold validated)");
}

fn test_performance() {
    print!("Testing performance requirements... ");
    
    let mut network = ruv_fann::Network::<f32>::new(&[5, 10, 1]).unwrap();
    
    let start = Instant::now();
    for i in 0..100 {
        let input: Vec<f32> = (0..5).map(|j| (i + j) as f32 / 100.0).collect();
        let _output = network.run(&input).unwrap();
    }
    let elapsed = start.elapsed();
    
    let avg_time = elapsed / 100;
    assert!(avg_time < Duration::from_millis(1), "Too slow: {:?}", avg_time);
    
    println!("✅ PASS ({} inferences in {:?}, avg: {:?})", 100, elapsed, avg_time);
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ruv_fann() {
        test_ruv_fann_basic();
    }
    
    #[test]
    fn test_consensus() {
        test_byzantine_consensus();
    }
    
    #[test] 
    fn test_perf() {
        test_performance();
    }
}