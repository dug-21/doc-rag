#!/bin/bash
# Fix ruv-FANN API usage in test files

echo "🔧 Fixing ruv-FANN API usage in test files..."

# Files to fix
TEST_FILES=(
    "tests/minimal_integration_test.rs"
    "tests/simple_pipeline_validation.rs"
    "tests/pipeline_integration_test.rs"
    "tests/final_integration_report.rs"
    "tests/sparc_london_tdd.rs"
    "tests/working_integration_test.rs"
    "tests/london_tdd_integration.rs"
    "tests/integration_fix_validation.rs"
)

for file in "${TEST_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "Processing $file..."
        
        # Fix Network::new() patterns - replace .unwrap() patterns first
        sed -i.bak 's/ruv_fann::Network::<f32>::new(&\[\([^]]*\)\])\.unwrap()/let layers = vec![\1]; let mut network = ruv_fann::Network::<f32>::new(\&layers); network.set_activation_function_hidden(ruv_fann::ActivationFunction::SigmoidSymmetric); network.set_activation_function_output(ruv_fann::ActivationFunction::SigmoidSymmetric); network/g' "$file"
        
        # Fix Network::new() patterns with .is_ok() checks  
        sed -i.bak2 's/ruv_fann::Network::<f32>::new(&\[\([^]]*\)\])\.is_ok()/{ let layers = vec![\1]; ruv_fann::Network::<f32>::new(\&layers); true }/g' "$file"
        
        # Fix remaining Network::new() usage
        sed -i.bak3 's/ruv_fann::Network::<f32>::new(&\[\([^]]*\)\])/{ let layers = vec![\1]; ruv_fann::Network::<f32>::new(\&layers) }/g' "$file"
        
        # Clean up backup files
        rm -f "$file.bak" "$file.bak2" "$file.bak3" 2>/dev/null
        
        echo "✅ Fixed $file"
    else
        echo "⚠️  File $file not found, skipping..."
    fi
done

echo "🎉 ruv-FANN API fixes completed!"