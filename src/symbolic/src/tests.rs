// src/symbolic/src/tests.rs
// Basic integration tests for symbolic reasoning module

#[cfg(test)]
mod tests {
    use crate::{DatalogEngine, PrologEngine, LogicParser};
    
    #[tokio::test]
    async fn test_symbolic_module_integration() {
        // Test that all main components can be instantiated
        let datalog_engine = DatalogEngine::new().await.expect("DatalogEngine initialization failed");
        assert!(datalog_engine.is_initialized().await);
        
        let prolog_engine = PrologEngine::new().await.expect("PrologEngine initialization failed");
        assert!(prolog_engine.is_initialized());
        
        let logic_parser = LogicParser::new().await.expect("LogicParser initialization failed");
        assert!(logic_parser.is_initialized());
    }
    
    #[tokio::test]
    async fn test_basic_requirement_compilation() {
        let requirement = "Cardholder data MUST be encrypted";
        
        let rule = DatalogEngine::compile_requirement_to_rule(requirement)
            .await
            .expect("Rule compilation failed");
        
        assert!(!rule.text.is_empty());
        assert_eq!(rule.rule_type, crate::types::RequirementType::Must);
        assert!(rule.text.contains("requires_encryption"));
        assert!(rule.text.contains("cardholder_data"));
    }
    
    #[tokio::test]
    async fn test_logic_parsing_basic() {
        let parser = LogicParser::new().await.expect("Parser initialization failed");
        
        let requirement = "Payment data MUST be encrypted when stored in databases";
        let parsed = parser.parse_requirement_to_logic(requirement)
            .await
            .expect("Parsing failed");
        
        assert_eq!(parsed.requirement_type, crate::types::RequirementType::Must);
        assert!(!parsed.subject.is_empty());
        assert!(!parsed.predicate.is_empty());
        assert!(!parsed.conditions.is_empty());
    }
}