#!/usr/bin/env python3
"""
Neo4j Graph Database Integration Validation Script for Phase 1
===============================================================

This script validates Neo4j connectivity, performance, and CONSTRAINT-002 compliance.
It tests graph traversal performance and section relationship creation as specified.
"""

import time
import asyncio
from neo4j import GraphDatabase
import logging
import json
from typing import Dict, List, Any
from datetime import datetime
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Neo4jValidator:
    """Validates Neo4j integration for Phase 1 requirements"""
    
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "neo4jpassword"):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self.validation_results = {
            'connectivity': {},
            'performance': {},
            'constraint_002': {},
            'section_relationships': {},
            'overall_status': 'UNKNOWN'
        }
    
    def connect(self):
        """Establish connection to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            logger.info(f"Successfully connected to Neo4j at {self.uri}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            self.validation_results['connectivity']['error'] = str(e)
            return False
    
    def test_basic_connectivity(self) -> Dict[str, Any]:
        """Test basic Neo4j connectivity and version"""
        results = {'status': 'FAILED', 'details': {}}
        
        try:
            with self.driver.session() as session:
                # Test basic query
                result = session.run("RETURN 1 as test, timestamp() as current_time")
                record = result.single()
                
                if record and record['test'] == 1:
                    results['status'] = 'PASSED'
                    results['details']['basic_query'] = True
                    results['details']['timestamp'] = record['current_time']
                    
                    # Get Neo4j version
                    version_result = session.run("CALL dbms.components() YIELD name, versions, edition")
                    version_record = version_result.single()
                    if version_record:
                        results['details']['version'] = version_record['versions'][0]
                        results['details']['edition'] = version_record['edition']
                        
                    logger.info(f"Basic connectivity test PASSED - Neo4j {results['details'].get('version', 'unknown')}")
                else:
                    results['details']['error'] = "Test query returned unexpected result"
                    
        except Exception as e:
            results['details']['error'] = str(e)
            logger.error(f"Basic connectivity test failed: {str(e)}")
            
        return results
    
    def setup_test_graph(self, session) -> int:
        """Create test data for performance validation"""
        logger.info("Setting up test graph with sample requirements and relationships...")
        
        # Clear existing test data
        session.run("MATCH (n) WHERE n.id STARTS WITH 'TEST-' DETACH DELETE n")
        
        # Create test requirements
        create_nodes_query = """
        UNWIND range(1, 50) as i
        CREATE (r:Requirement {
            id: 'TEST-REQ-' + toString(i),
            text: 'Test requirement ' + toString(i) + ' for performance validation',
            section: 'TEST-SEC-' + toString((i-1) / 10 + 1),
            requirement_type: CASE i % 3 
                WHEN 0 THEN 'MAY'
                WHEN 1 THEN 'SHOULD'
                ELSE 'MUST'
            END,
            domain: CASE i % 4
                WHEN 0 THEN 'encryption'
                WHEN 1 THEN 'access_control'
                WHEN 2 THEN 'monitoring'
                ELSE 'compliance'
            END,
            priority: CASE i % 4
                WHEN 0 THEN 'CRITICAL'
                WHEN 1 THEN 'HIGH'
                WHEN 2 THEN 'MEDIUM'
                ELSE 'LOW'
            END,
            created_at: datetime()
        })
        """
        result = session.run(create_nodes_query)
        nodes_created = result.consume().counters.nodes_created
        
        # Create test relationships for traversal
        create_relationships_query = """
        MATCH (r1:Requirement), (r2:Requirement)
        WHERE r1.id STARTS WITH 'TEST-REQ-' AND r2.id STARTS WITH 'TEST-REQ-'
        AND toInteger(split(r1.id, '-')[2]) < toInteger(split(r2.id, '-')[2])
        AND toInteger(split(r2.id, '-')[2]) - toInteger(split(r1.id, '-')[2]) <= 3
        AND rand() < 0.3
        CREATE (r1)-[:REFERENCES]->(r2)
        """
        result = session.run(create_relationships_query)
        relationships_created = result.consume().counters.relationships_created
        
        # Create some DEPENDS_ON relationships
        create_dependencies_query = """
        MATCH (r1:Requirement), (r2:Requirement)
        WHERE r1.id STARTS WITH 'TEST-REQ-' AND r2.id STARTS WITH 'TEST-REQ-'
        AND toInteger(split(r1.id, '-')[2]) < toInteger(split(r2.id, '-')[2])
        AND toInteger(split(r2.id, '-')[2]) - toInteger(split(r1.id, '-')[2]) = 5
        CREATE (r1)-[:DEPENDS_ON]->(r2)
        """
        result = session.run(create_dependencies_query)
        dependencies_created = result.consume().counters.relationships_created
        
        logger.info(f"Created {nodes_created} test nodes and {relationships_created + dependencies_created} relationships")
        return nodes_created
    
    def test_performance_constraints(self) -> Dict[str, Any]:
        """Test CONSTRAINT-002: <200ms graph traversal for 3-hop queries"""
        results = {
            'status': 'FAILED',
            'constraint_002_compliance': False,
            'performance_metrics': {},
            'details': {}
        }
        
        try:
            with self.driver.session() as session:
                # Setup test data
                node_count = self.setup_test_graph(session)
                results['details']['test_nodes_created'] = node_count
                
                # Test 3-hop traversal queries multiple times
                traversal_query = """
                MATCH path = (start:Requirement {id: $start_id})
                    -[:REFERENCES|DEPENDS_ON*1..3]-(related:Requirement)
                WHERE related.id STARTS WITH 'TEST-REQ-'
                RETURN path, 
                       start,
                       related,
                       length(path) as depth,
                       [r in relationships(path) | type(r)] as relationship_chain
                ORDER BY depth, related.id
                LIMIT 100
                """
                
                execution_times = []
                successful_queries = 0
                total_results = 0
                
                # Run multiple iterations to test consistency
                for i in range(10):
                    start_id = f'TEST-REQ-{(i % 5) + 1}'
                    
                    start_time = time.time()
                    result = session.run(traversal_query, start_id=start_id)
                    records = list(result)
                    end_time = time.time()
                    
                    execution_time_ms = (end_time - start_time) * 1000
                    execution_times.append(execution_time_ms)
                    total_results += len(records)
                    
                    if execution_time_ms < 200:
                        successful_queries += 1
                        
                    logger.info(f"3-hop traversal {i+1}: {execution_time_ms:.2f}ms, {len(records)} results")
                
                # Calculate performance metrics
                avg_time = statistics.mean(execution_times)
                max_time = max(execution_times)
                min_time = min(execution_times)
                
                results['performance_metrics'] = {
                    'average_execution_time_ms': round(avg_time, 2),
                    'max_execution_time_ms': round(max_time, 2),
                    'min_execution_time_ms': round(min_time, 2),
                    'total_queries': len(execution_times),
                    'successful_queries_under_200ms': successful_queries,
                    'success_rate': round((successful_queries / len(execution_times)) * 100, 2),
                    'total_results_returned': total_results,
                    'all_execution_times_ms': [round(t, 2) for t in execution_times]
                }
                
                # Check CONSTRAINT-002 compliance
                constraint_002_met = max_time < 200
                results['constraint_002_compliance'] = constraint_002_met
                
                if constraint_002_met and successful_queries == len(execution_times):
                    results['status'] = 'PASSED'
                    logger.info(f"CONSTRAINT-002 PASSED: All 3-hop queries completed in <200ms (max: {max_time:.2f}ms)")
                else:
                    results['status'] = 'FAILED'
                    results['details']['failure_reason'] = f"Max execution time {max_time:.2f}ms exceeds 200ms limit"
                    logger.warning(f"CONSTRAINT-002 FAILED: Max execution time {max_time:.2f}ms exceeds 200ms limit")
                
        except Exception as e:
            results['details']['error'] = str(e)
            logger.error(f"Performance test failed: {str(e)}")
            
        return results
    
    def test_section_relationships(self) -> Dict[str, Any]:
        """Test section relationship graph creation"""
        results = {'status': 'FAILED', 'details': {}}
        
        try:
            with self.driver.session() as session:
                # Create test document hierarchy
                create_document_query = """
                CREATE (doc:Document {
                    id: 'TEST-DOC-001',
                    title: 'Test Compliance Document',
                    version: '1.0',
                    doc_type: 'PCI-DSS',
                    created_at: datetime()
                })
                WITH doc
                CREATE (sec1:Section {
                    id: 'TEST-SEC-3.2',
                    number: '3.2',
                    title: 'Data Protection Requirements',
                    section_type: 'Requirements'
                }),
                (sec2:Section {
                    id: 'TEST-SEC-3.3',
                    number: '3.3',
                    title: 'Encryption Requirements',
                    section_type: 'Requirements'
                })
                WITH doc, sec1, sec2
                CREATE (doc)-[:CONTAINS]->(sec1),
                       (doc)-[:CONTAINS]->(sec2)
                WITH doc, sec1, sec2
                CREATE (req1:Requirement {
                    id: 'TEST-REQ-3.2.1',
                    text: 'Cardholder data must be encrypted at rest',
                    section: 'TEST-SEC-3.2',
                    requirement_type: 'MUST',
                    domain: 'encryption',
                    priority: 'HIGH',
                    created_at: datetime()
                }),
                (req2:Requirement {
                    id: 'TEST-REQ-3.3.1',
                    text: 'Encryption keys must be managed securely',
                    section: 'TEST-SEC-3.3',
                    requirement_type: 'MUST',
                    domain: 'encryption',
                    priority: 'HIGH',
                    created_at: datetime()
                })
                WITH doc, sec1, sec2, req1, req2
                CREATE (sec1)-[:CONTAINS]->(req1),
                       (sec2)-[:CONTAINS]->(req2),
                       (req1)-[:REFERENCES]->(req2)
                RETURN doc, sec1, sec2, req1, req2
                """
                
                start_time = time.time()
                result = session.run(create_document_query)
                records = list(result)
                creation_time = (time.time() - start_time) * 1000
                
                if records:
                    results['status'] = 'PASSED'
                    results['details']['hierarchy_created'] = True
                    results['details']['creation_time_ms'] = round(creation_time, 2)
                    results['details']['nodes_in_hierarchy'] = len(records)
                    
                    # Verify the hierarchy with a traversal query
                    verify_query = """
                    MATCH (doc:Document {id: 'TEST-DOC-001'})
                        -[:CONTAINS]->(sec:Section)
                        -[:CONTAINS]->(req:Requirement)
                    RETURN doc.title as document_title,
                           sec.title as section_title,
                           req.text as requirement_text,
                           sec.id as section_id,
                           req.id as requirement_id
                    """
                    
                    verify_result = session.run(verify_query)
                    hierarchy_records = list(verify_result)
                    
                    results['details']['hierarchy_verification'] = {
                        'requirements_in_sections': len(hierarchy_records),
                        'hierarchy_intact': len(hierarchy_records) >= 2
                    }
                    
                    logger.info(f"Section relationship test PASSED: Created hierarchy in {creation_time:.2f}ms")
                else:
                    results['details']['error'] = "No records returned from hierarchy creation"
                    
        except Exception as e:
            results['details']['error'] = str(e)
            logger.error(f"Section relationship test failed: {str(e)}")
            
        return results
    
    def test_schema_compliance(self) -> Dict[str, Any]:
        """Test Neo4j schema setup and constraints"""
        results = {'status': 'FAILED', 'details': {}}
        
        try:
            with self.driver.session() as session:
                # Check if constraints exist
                constraints_query = "SHOW CONSTRAINTS"
                constraints_result = session.run(constraints_query)
                constraints = list(constraints_result)
                
                # Check if indexes exist  
                indexes_query = "SHOW INDEXES"
                indexes_result = session.run(indexes_query)
                indexes = list(indexes_result)
                
                results['details']['constraints_count'] = len(constraints)
                results['details']['indexes_count'] = len(indexes)
                results['details']['schema_objects'] = {
                    'constraints': [dict(record) for record in constraints],
                    'indexes': [dict(record) for record in indexes]
                }
                
                if len(constraints) > 0 or len(indexes) > 0:
                    results['status'] = 'PASSED'
                    logger.info(f"Schema compliance PASSED: {len(constraints)} constraints, {len(indexes)} indexes")
                else:
                    results['status'] = 'PARTIAL'
                    results['details']['warning'] = "No constraints or indexes found - schema not fully initialized"
                    logger.warning("Schema compliance PARTIAL: No constraints or indexes found")
                
        except Exception as e:
            results['details']['error'] = str(e)
            logger.error(f"Schema compliance test failed: {str(e)}")
            
        return results
    
    def cleanup_test_data(self):
        """Clean up test data created during validation"""
        try:
            with self.driver.session() as session:
                cleanup_query = "MATCH (n) WHERE n.id STARTS WITH 'TEST-' DETACH DELETE n"
                result = session.run(cleanup_query)
                deleted_count = result.consume().counters.nodes_deleted
                logger.info(f"Cleaned up {deleted_count} test nodes")
        except Exception as e:
            logger.warning(f"Cleanup failed: {str(e)}")
    
    def run_validation(self) -> Dict[str, Any]:
        """Run complete Neo4j validation suite"""
        logger.info("Starting Neo4j Phase 1 Integration Validation")
        logger.info("=" * 60)
        
        if not self.connect():
            self.validation_results['overall_status'] = 'CONNECTION_FAILED'
            return self.validation_results
        
        try:
            # Test 1: Basic Connectivity
            logger.info("1. Testing basic connectivity...")
            self.validation_results['connectivity'] = self.test_basic_connectivity()
            
            # Test 2: Performance and CONSTRAINT-002
            logger.info("2. Testing performance constraints (CONSTRAINT-002)...")
            self.validation_results['constraint_002'] = self.test_performance_constraints()
            
            # Test 3: Section Relationships
            logger.info("3. Testing section relationship graph creation...")
            self.validation_results['section_relationships'] = self.test_section_relationships()
            
            # Test 4: Schema Compliance
            logger.info("4. Testing schema compliance...")
            schema_results = self.test_schema_compliance()
            self.validation_results['schema'] = schema_results
            
            # Determine overall status
            all_tests = [
                self.validation_results['connectivity']['status'],
                self.validation_results['constraint_002']['status'],
                self.validation_results['section_relationships']['status'],
                schema_results['status']
            ]
            
            if all(status == 'PASSED' for status in all_tests):
                self.validation_results['overall_status'] = 'PASSED'
            elif any(status == 'PASSED' for status in all_tests):
                self.validation_results['overall_status'] = 'PARTIAL'
            else:
                self.validation_results['overall_status'] = 'FAILED'
            
            # Clean up test data
            self.cleanup_test_data()
            
        except Exception as e:
            logger.error(f"Validation suite failed: {str(e)}")
            self.validation_results['validation_error'] = str(e)
            self.validation_results['overall_status'] = 'ERROR'
        
        finally:
            if self.driver:
                self.driver.close()
        
        # Add summary
        self.validation_results['validation_completed_at'] = datetime.now().isoformat()
        self.validation_results['summary'] = self.generate_summary()
        
        return self.validation_results
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate validation summary"""
        summary = {
            'overall_status': self.validation_results['overall_status'],
            'tests_run': 0,
            'tests_passed': 0,
            'critical_issues': [],
            'recommendations': []
        }
        
        # Count tests
        test_results = [
            self.validation_results.get('connectivity', {}).get('status'),
            self.validation_results.get('constraint_002', {}).get('status'),
            self.validation_results.get('section_relationships', {}).get('status'),
            self.validation_results.get('schema', {}).get('status')
        ]
        
        summary['tests_run'] = len([t for t in test_results if t])
        summary['tests_passed'] = len([t for t in test_results if t == 'PASSED'])
        
        # Check for critical issues
        constraint_002 = self.validation_results.get('constraint_002', {})
        if not constraint_002.get('constraint_002_compliance', False):
            summary['critical_issues'].append('CONSTRAINT-002 not met: 3-hop graph traversal exceeds 200ms limit')
        
        if self.validation_results.get('connectivity', {}).get('status') != 'PASSED':
            summary['critical_issues'].append('Neo4j connectivity issues detected')
        
        # Add recommendations
        if constraint_002.get('performance_metrics', {}).get('max_execution_time_ms', 0) > 150:
            summary['recommendations'].append('Consider adding additional indexes to improve query performance')
        
        schema_status = self.validation_results.get('schema', {}).get('status')
        if schema_status in ['PARTIAL', 'FAILED']:
            summary['recommendations'].append('Initialize Neo4j schema with proper constraints and indexes')
        
        return summary

def main():
    """Main validation function"""
    print("Neo4j Graph Database Integration Validation")
    print("Phase 1 Requirement Validation Suite")
    print("=" * 50)
    
    validator = Neo4jValidator()
    results = validator.run_validation()
    
    # Print results
    print("\n" + "=" * 50)
    print("VALIDATION RESULTS")
    print("=" * 50)
    
    print(f"Overall Status: {results['overall_status']}")
    print(f"Validation Completed: {results['validation_completed_at']}")
    
    summary = results['summary']
    print(f"\nTest Summary: {summary['tests_passed']}/{summary['tests_run']} tests passed")
    
    if summary['critical_issues']:
        print(f"\n❌ Critical Issues ({len(summary['critical_issues'])}):")
        for issue in summary['critical_issues']:
            print(f"   - {issue}")
    
    if summary['recommendations']:
        print(f"\n💡 Recommendations ({len(summary['recommendations'])}):")
        for rec in summary['recommendations']:
            print(f"   - {rec}")
    
    print("\n" + "=" * 50)
    print("DETAILED RESULTS")
    print("=" * 50)
    
    # Connectivity Results
    conn = results['connectivity']
    print(f"\n1. Connectivity Test: {conn.get('status', 'N/A')}")
    if conn.get('details'):
        details = conn['details']
        if 'version' in details:
            print(f"   Neo4j Version: {details['version']} ({details.get('edition', 'unknown')})")
        if 'error' in details:
            print(f"   Error: {details['error']}")
    
    # CONSTRAINT-002 Results  
    constraint = results['constraint_002']
    print(f"\n2. CONSTRAINT-002 Compliance: {'✅ PASSED' if constraint.get('constraint_002_compliance') else '❌ FAILED'}")
    if constraint.get('performance_metrics'):
        metrics = constraint['performance_metrics']
        print(f"   Average 3-hop query time: {metrics['average_execution_time_ms']}ms")
        print(f"   Maximum 3-hop query time: {metrics['max_execution_time_ms']}ms")
        print(f"   Success rate: {metrics['success_rate']}%")
        print(f"   Total queries tested: {metrics['total_queries']}")
    
    # Section Relationships Results
    sections = results['section_relationships']
    print(f"\n3. Section Relationship Test: {sections.get('status', 'N/A')}")
    if sections.get('details'):
        details = sections['details']
        if 'creation_time_ms' in details:
            print(f"   Hierarchy creation time: {details['creation_time_ms']}ms")
        if 'hierarchy_verification' in details:
            verification = details['hierarchy_verification']
            print(f"   Requirements in sections: {verification['requirements_in_sections']}")
            print(f"   Hierarchy intact: {verification['hierarchy_intact']}")
    
    # Schema Results
    schema = results.get('schema', {})
    print(f"\n4. Schema Compliance: {schema.get('status', 'N/A')}")
    if schema.get('details'):
        details = schema['details']
        print(f"   Constraints found: {details.get('constraints_count', 0)}")
        print(f"   Indexes found: {details.get('indexes_count', 0)}")
    
    # Save results to file
    with open('neo4j_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: neo4j_validation_results.json")
    
    return results['overall_status'] == 'PASSED'

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)