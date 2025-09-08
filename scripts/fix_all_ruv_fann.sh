#!/bin/bash

# Comprehensive script to fix all ruv-FANN API usage issues

echo "=== Fixing all ruv-FANN compilation errors ==="

# Find all test files with Network usage
TEST_FILES=(
    "tests/working_integration_test.rs"
    "tests/final_integration_report.rs"
    "tests/simple_pipeline_validation.rs"
    "tests/minimal_integration_test.rs"
    "tests/integration_fix_validation.rs"
    "tests/sparc_london_tdd.rs"
    "standalone_test.rs"
)

echo "Step 1: Fix Network::new() calls in test files..."
for file in "${TEST_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "Fixing $file..."
        
        # Fix Network::new() patterns - handle Result properly
        sed -i.bak 's/let mut network = ruv_fann::Network::<f32>::new(&layers);/let mut network = ruv_fann::Network::<f32>::new(\&layers)?;/g' "$file"
        sed -i.bak2 's/let network = ruv_fann::Network::<f32>::new(&layers);/let network = ruv_fann::Network::<f32>::new(\&layers)?;/g' "$file"
        
        # Fix network.run() patterns - handle Result properly
        sed -i.bak3 's/network\.run(&[^)]*).unwrap()/network.run(\&input)?/g' "$file"
        sed -i.bak4 's/let _output = network\.run(&[^)]*).unwrap();/let _output = network.run(\&input)?;/g' "$file"
        sed -i.bak5 's/let neural_result = network\.run(&[^)]*).unwrap();/let neural_result = network.run(\&neural_input)?;/g' "$file"
        sed -i.bak6 's/let neural_output = network\.run(&[^)]*).unwrap();/let neural_output = network.run(\&neural_input)?;/g' "$file"
        
        # Fix .is_ok()/.is_err() patterns on network.run()
        sed -i.bak7 's/network\.run(&[^)]*).is_ok()/{ let _result = network.run(\&input)?; true }/g' "$file"
        sed -i.bak8 's/network\.run(&[^)]*).is_err()/{ let result = network.run(\&input); result.is_err() }/g' "$file"
        
        # Clean up backup files
        rm -f "$file".bak*
        
        echo "  ✓ Fixed $file"
    fi
done

echo "Step 2: Fix chunker files..."
CHUNKER_FILES=(
    "src/chunker/src/neural_trainer.rs"
    "src/chunker/src/neural_chunker.rs"
    "src/chunker/src/neural_chunker_working.rs"
    "src/chunker/src/simple_lib.rs"
)

for file in "${CHUNKER_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "Fixing $file..."
        
        # Fix Network::new() patterns
        sed -i.bak 's/let mut network = Network::new(&layers);/let mut network = Network::new(\&layers)?;/g' "$file"
        sed -i.bak2 's/let mut best_network = Network::new(&layers);/let mut best_network = Network::new(\&layers)?;/g' "$file"
        
        # Fix network.run() patterns
        sed -i.bak3 's/let output = network\.run(input);/let output = network.run(input)?;/g' "$file"
        sed -i.bak4 's/let _output = network\.run(&input);/let _output = network.run(\&input)?;/g' "$file"
        
        # Clean up backup files
        rm -f "$file".bak*
        
        echo "  ✓ Fixed $file"
    fi
done

echo "=== All ruv-FANN fixes applied ==="
echo "Please run 'cargo check' to verify the fixes."