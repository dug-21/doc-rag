# Phase 3: Simple Validation Setup
## Minimal Docker Deployment for Testing

**Goal**: Get the system running TODAY for validation, not in 8 weeks for production.

---

## 🎯 Core Objectives (Simplified)

1. **Run the system in Docker** - MongoDB + API, that's it
2. **Test with real PDFs** - Upload PDF, ask questions, get answers
3. **Regression testing** - Generate tests, compare results
4. **Model training** - Feed Q&A pairs, improve accuracy

**NO**: Kubernetes, React UI, WebSockets, Service Mesh, fancy monitoring  
**YES**: docker-compose, curl commands, JSON responses, simple scripts

---

## 🐳 Minimal Docker Setup

### Just 3 Containers:
```yaml
# docker-compose.yml
services:
  mongodb:
    image: mongo:7.0
    ports: ["27017:27017"]
    volumes: ["./data/mongo:/data/db"]
  
  redis:
    image: redis:7.2-alpine
    ports: ["6379:6379"]
  
  api:
    build: .
    ports: ["8080:8080"]
    environment:
      - MONGODB_URL=mongodb://mongodb:27017
      - REDIS_URL=redis://redis:6379
    volumes: ["./models:/models"]
```

**Start everything**: `docker-compose up`  
**That's it!**

---

## 📝 Pipeline 1: Simple Regression Testing

### One Python script:
```bash
# test_regression.py
# 1. Generate 10 test PDFs with known Q&A
# 2. Upload each PDF
# 3. Ask standard questions
# 4. Compare answers to baseline
# 5. Output: PASS/FAIL + accuracy score

python test_regression.py --baseline baseline.json --output results.json
```

**Time to implement**: 1 day

---

## 📄 Pipeline 2: PDF Upload & Query (No UI!)

### Two REST endpoints:
```bash
# Upload PDF
curl -X POST http://localhost:8080/upload \
  -F "file=@mydocument.pdf" \
  -F "name=mydoc"

# Response: {"id": "doc_123", "status": "processed"}

# Ask question
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "doc_123", "question": "What are the main points?"}'

# Response: {
#   "answer": "The main points are...",
#   "citations": [...],
#   "confidence": 0.95
# }
```

**Time to implement**: 1 day

---

## 🧠 Pipeline 3: Simple Model Training

### One training script:
```bash
# train_model.py
# Input: qa_pairs.csv with columns [question, answer, context]
# Output: new model file

python train_model.py \
  --input qa_pairs.csv \
  --model models/model_v2.bin \
  --epochs 10

# Deploy: just copy the model file
cp models/model_v2.bin models/current.bin
docker-compose restart api
```

**Time to implement**: 1 day

---

## 🚀 Quick Start Guide

### Day 1: Get it running
```bash
# 1. Build the Docker image
docker build -t doc-rag-api .

# 2. Start services
docker-compose up -d

# 3. Test it works
curl http://localhost:8080/health
```

### Day 2: Test with PDFs
```bash
# Upload a PDF
./scripts/upload_pdf.sh sample.pdf

# Ask questions
./scripts/query.sh "What is this document about?"
```

### Day 3: Run regression tests
```bash
# Generate and test
python test_regression.py

# Check results
cat results.json | jq '.accuracy'
```

---

## 📊 Success Metrics (Simple!)

- ✅ System runs in Docker
- ✅ Can upload PDF and get answers
- ✅ Accuracy >= 90% on test set
- ✅ Response time < 2 seconds
- ✅ Regression tests pass

**NOT tracking**: Kubernetes metrics, distributed tracing, A/B tests, etc.

---

## 📁 File Structure (Minimal)
```
doc-rag/
├── docker-compose.yml      # 20 lines
├── Dockerfile             # Standard Rust multi-stage
├── scripts/
│   ├── upload_pdf.sh     # 5 lines
│   ├── query.sh          # 5 lines
│   └── test_all.sh       # 10 lines
├── tests/
│   ├── test_regression.py # 200 lines
│   ├── train_model.py     # 150 lines
│   └── test_data/         # Sample PDFs
└── models/
    └── current.bin        # Active model
```

---

## ⏱️ Timeline: 3 DAYS (not 8 weeks!)

**Day 1**: Docker setup + basic API  
**Day 2**: Testing scripts  
**Day 3**: Training script + validation  

**Total effort**: 1 developer, 3 days

---

## 💰 Cost: ~$0 (run locally)

- Local Docker: Free
- MongoDB: Free (local)
- Redis: Free (local)
- No cloud services needed for validation

---

## 🎯 This is ALL you need to:

1. ✅ Validate the system works
2. ✅ Test with real PDFs
3. ✅ Measure accuracy
4. ✅ Improve with training
5. ✅ Run regression tests

**AFTER validation works**, THEN consider production architecture.

---

## Next Step: Just run these commands:

```bash
# 1. Create docker-compose.yml (copy from above)
# 2. Build and start
docker-compose up

# 3. Test
curl -X POST http://localhost:8080/upload -F "file=@test.pdf"
curl -X POST http://localhost:8080/query -d '{"question":"test"}'

# Done! System validated.
```