# Dockerfile for QuantumDynamics Pro API
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash quantum
RUN chown -R quantum:quantum /app
USER quantum

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "quantum_api_service.py", "--host", "0.0.0.0", "--port", "8000"]

---
# requirements.txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
numpy==1.24.3
scipy==1.11.4
pydantic==2.5.0
prometheus-client==0.19.0
python-multipart==0.0.6

---
# docker-compose.yml
version: '3.8'

services:
  quantum-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=info
      - WORKERS=4
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=quantum2024
    volumes:
      - grafana-storage:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources

volumes:
  grafana-storage:

---
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'quantum-api'
    static_configs:
      - targets: ['quantum-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

---
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quantum-api
  labels:
    app: quantum-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: quantum-api
  template:
    metadata:
      labels:
        app: quantum-api
    spec:
      containers:
      - name: quantum-api
        image: quantumdynamics/quantum-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: WORKERS
          value: "1"
        - name: LOG_LEVEL
          value: "info"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: quantum-api-service
spec:
  selector:
    app: quantum-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer

---
# Client Examples

# Python client example
import requests
import json

class QuantumAPIClient:
    """Client for QuantumDynamics Pro API"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self):
        """Check API health"""
        response = self.session.get(f"{self.base_url}/health")
        return response.json()
    
    def calculate_commutator(self, H_matrix, O_matrix):
        """Calculate commutator [H, O]"""
        payload = {
            "H": {"matrix": H_matrix},
            "O": {"matrix": O_matrix}
        }
        response = self.session.post(
            f"{self.base_url}/quantum/commutator",
            json=payload
        )
        return response.json()
    
    def evolve_dyson_series(self, hamiltonian_type, parameters, evolution_time, target_error=1e-12):
        """Calculate Dyson series evolution"""
        payload = {
            "hamiltonian": {
                "type": hamiltonian_type,
                "parameters": parameters
            },
            "evolution_time": evolution_time,
            "target_error": target_error
        }
        response = self.session.post(
            f"{self.base_url}/quantum/dyson-series",
            json=payload
        )
        return response.json()
    
    def heisenberg_evolution(self, initial_observable, hamiltonian_type, parameters, evolution_time, time_steps=100):
        """Calculate Heisenberg evolution"""
        payload = {
            "initial_observable": {"matrix": initial_observable},
            "hamiltonian": {
                "type": hamiltonian_type,
                "parameters": parameters
            },
            "evolution_time": evolution_time,
            "time_steps": time_steps
        }
        response = self.session.post(
            f"{self.base_url}/quantum/heisenberg-evolution",
            json=payload
        )
        return response.json()

# Example usage
if __name__ == "__main__":
    client = QuantumAPIClient()
    
    # Test 1: Health check
    print("🏥 Health Check:")
    health = client.health_check()
    print(f"Status: {health['status']}")
    print(f"Precision: {health.get('precision_check', 'unknown')}")
    
    # Test 2: Commutator calculation
    print("\n🔬 Commutator [σ_x, σ_z]:")
    sigma_x = [[0, 1], [1, 0]]
    sigma_z = [[1, 0], [0, -1]]
    
    result = client.calculate_commutator(sigma_x, sigma_z)
    print(f"Calculation ID: {result['calculation_id']}")
    print(f"Computation time: {result['computation_time']:.4f}s")
    print(f"Norm: {result['norm']:.6f}")
    
    # Test 3: Driven qubit evolution
    print("\n⚡ Driven Qubit Evolution:")
    evolution_result = client.evolve_dyson_series(
        hamiltonian_type="driven_qubit",
        parameters={"omega_0": 1.0, "omega_d": 0.98, "rabi_freq": 0.01},
        evolution_time=1.0,
        target_error=1e-12
    )
    print(f"Unitarity error: {evolution_result['unitarity_error']:.2e}")
    print(f"Target achieved: {evolution_result['target_achieved']}")
    
    # Test 4: Heisenberg evolution
    print("\n🌀 Heisenberg Evolution:")
    heisenberg_result = client.heisenberg_evolution(
        initial_observable=sigma_x,
        hamiltonian_type="rabi",
        parameters={"omega_0": 0.5, "rabi_freq": 0.1},
        evolution_time=2.0
    )
    print(f"Norm preservation: {heisenberg_result['norm_preservation']:.2e}")
    print(f"Time points: {len(heisenberg_result['time_points'])}")

---
# JavaScript/Node.js client example
const axios = require('axios');

class QuantumAPIClient {
    constructor(baseURL = 'http://localhost:8000') {
        this.client = axios.create({
            baseURL,
            timeout: 30000,
            headers: {
                'Content-Type': 'application/json'
            }
        });
    }

    async healthCheck() {
        const response = await this.client.get('/health');
        return response.data;
    }

    async calculateCommutator(HMatrix, OMatrix) {
        const payload = {
            H: { matrix: HMatrix },
            O: { matrix: OMatrix }
        };
        const response = await this.client.post('/quantum/commutator', payload);
        return response.data;
    }

    async evolveDysonSeries(hamiltonianType, parameters, evolutionTime, targetError = 1e-12) {
        const payload = {
            hamiltonian: {
                type: hamiltonianType,
                parameters
            },
            evolution_time: evolutionTime,
            target_error: targetError
        };
        const response = await this.client.post('/quantum/dyson-series', payload);
        return response.data;
    }
}

// Example usage
async function runExample() {
    const client = new QuantumAPIClient();
    
    try {
        // Health check
        console.log('🏥 Health Check:');
        const health = await client.healthCheck();
        console.log(`Status: ${health.status}`);
        
        // Commutator calculation
        console.log('\n🔬 Commutator Calculation:');
        const sigmaX = [[0, 1], [1, 0]];
        const sigmaZ = [[1, 0], [0, -1]];
        
        const commutatorResult = await client.calculateCommutator(sigmaX, sigmaZ);
        console.log(`Computation time: ${commutatorResult.computation_time.toFixed(4)}s`);
        console.log(`Norm: ${commutatorResult.norm.toFixed(6)}`);
        
        // Dyson series
        console.log('\n⚡ Dyson Series Evolution:');
        const evolutionResult = await client.evolveDysonSeries(
            'driven_qubit',
            { omega_0: 1.0, omega_d: 0.98, rabi_freq: 0.01 },
            1.0
        );
        console.log(`Unitarity error: ${evolutionResult.unitarity_error.toExponential(2)}`);
        console.log(`Target achieved: ${evolutionResult.target_achieved}`);
        
    } catch (error) {
        console.error('Error:', error.response?.data || error.message);
    }
}

runExample();

---
# Deployment scripts

# deploy.sh
#!/bin/bash
set -e

echo "🚀 Deploying QuantumDynamics Pro API"

# Build Docker image
echo "📦 Building Docker image..."
docker build -t quantumdynamics/quantum-api:latest .

# Run tests
echo "🧪 Running tests..."
docker run --rm quantumdynamics/quantum-api:latest python -m pytest tests/

# Deploy with docker-compose
echo "🌐 Deploying services..."
docker-compose up -d

# Wait for services to be ready
echo "⏱️  Waiting for services to start..."
sleep 30

# Health check
echo "🏥 Performing health check..."
curl -f http://localhost:8000/health || exit 1

echo "✅ Deployment complete!"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "📊 Prometheus: http://localhost:9090"
echo "📈 Grafana: http://localhost:3000 (admin/quantum2024)"

# k8s-deploy.sh
#!/bin/bash
set -e

echo "☸️  Deploying to Kubernetes"

# Apply Kubernetes manifests
kubectl apply -f kubernetes/

# Wait for deployment
kubectl rollout status deployment/quantum-api

# Get service endpoint
kubectl get service quantum-api-service

echo "✅ Kubernetes deployment complete!"

---
# Performance testing script

# load_test.py
import asyncio
import aiohttp
import time
import statistics

async def test_endpoint(session, url, payload):
    """Test a single endpoint"""
    start_time = time.time()
    async with session.post(url, json=payload) as response:
        await response.json()
        return time.time() - start_time

async def load_test():
    """Run load test on quantum API"""
    base_url = "http://localhost:8000"
    concurrent_requests = 50
    total_requests = 1000
    
    # Test payload
    payload = {
        "H": {"matrix": [[0, 1], [1, 0]]},
        "O": {"matrix": [[1, 0], [0, -1]]}
    }
    
    async with aiohttp.ClientSession() as session:
        print(f"🚀 Starting load test: {total_requests} requests, {concurrent_requests} concurrent")
        
        start_time = time.time()
        
        # Create semaphore for concurrent requests
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def bounded_test():
            async with semaphore:
                return await test_endpoint(
                    session, 
                    f"{base_url}/quantum/commutator", 
                    payload
                )
        
        # Run all requests
        tasks = [bounded_test() for _ in range(total_requests)]
        response_times = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        avg_response_time = statistics.mean(response_times)
        median_response_time = statistics.median(response_times)
        p95_response_time = sorted(response_times)[int(0.95 * len(response_times))]
        requests_per_second = total_requests / total_time
        
        print(f"\n📊 Load Test Results:")
        print(f"Total requests: {total_requests}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Requests/second: {requests_per_second:.2f}")
        print(f"Average response time: {avg_response_time*1000:.2f}ms")
        print(f"Median response time: {median_response_time*1000:.2f}ms")
        print(f"95th percentile: {p95_response_time*1000:.2f}ms")

if __name__ == "__main__":
    asyncio.run(load_test())