// Mock Services Orchestrator
// Starts all mock services for isolated testing
const { spawn } = require('child_process');
const path = require('path');

const services = [
  { name: 'API Mock', script: 'api-mock.js', port: 3001 },
  { name: 'Storage Mock', script: 'storage-mock.js', port: 3002 },
  { name: 'Embedder Mock', script: 'embedder-mock.js', port: 3003 },
  { name: 'Integration Mock', script: 'integration-mock.js', port: 3004 }
];

console.log('🚀 Starting Doc-RAG Mock Services...\n');

// Start each service
services.forEach(service => {
  const child = spawn('node', [path.join(__dirname, service.script)], {
    stdio: 'inherit',
    env: { ...process.env, PORT: service.port }
  });

  child.on('error', (error) => {
    console.error(`❌ Failed to start ${service.name}: ${error.message}`);
  });

  console.log(`✅ ${service.name} starting on port ${service.port}`);
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\n🛑 Shutting down mock services...');
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log('\n🛑 Shutting down mock services...');
  process.exit(0);
});

console.log('\n📝 Mock services are running. Press Ctrl+C to stop.\n');
console.log('Health check endpoints:');
services.forEach(service => {
  console.log(`   ${service.name}: http://localhost:${service.port}/health`);
});
console.log();