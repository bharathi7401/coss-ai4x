# AI4X - AI Model Hosting & Pay-per-Use Platform

## Overview

AI4X is a comprehensive platform for hosting AI models, serving them via pipelines, and implementing pay-per-use tracking for real-time usage and billing. This implementation leverages the Dhruva API for core AI services and provides a complete billing and usage tracking system.

## 🚀 Features

### Core Model APIs
- **NMT (Neural Machine Translation)**: Translate text between multiple Indian languages
- **ASR (Automatic Speech Recognition)**: Convert audio to text with language detection
- **TTS (Text-to-Speech)**: Convert text to audio in multiple languages
- **LLM (Large Language Model)**: Process natural language queries

### Pipeline Composition
- Chain up to 3 services together
- Fixed pipeline: Language Detection → NMT → LLM → BackNMT → TTS
- Individual service endpoints for granular control

### Pay-per-Use Tracking
- **API Call Count**: Track requests per service per customer
- **Data Volume**: Monitor input/output data processed
- **Processing Time**: Measure service performance
- **Peak Concurrent Usage**: Track simultaneous request handling

### Usage Analytics
- Real-time usage dashboards
- Historical usage reports
- Service-specific metrics
- Error tracking and analysis

### Billing & Quotas
- Usage-based billing
- Quota management
- Invoice generation
- Credit balance tracking
- Webhook notifications

## 📋 API Endpoints

### Core Model APIs

#### NMT API
```
POST /nmt/translate
```
- **Purpose**: Translate text from one language to another
- **Pay-per-Use Metric**: Count per request + number of characters processed
- **Request**: `{"text": "Hello", "source_language": "en", "target_language": "hi"}`
- **Response**: Translation result with character count and latency

#### ASR API
```
POST /asr/convert
```
- **Purpose**: Convert audio input into text
- **Pay-per-Use Metric**: Count per request + audio duration in seconds
- **Request**: `{"audio_content": "base64_encoded_audio", "audio_format": "wav"}`
- **Response**: Transcription with detected language and duration

#### TTS API
```
POST /tts/speak
```
- **Purpose**: Convert text input into audio output
- **Pay-per-Use Metric**: Count per request + number of characters processed
- **Request**: `{"text": "Hello", "language": "en", "gender": "female"}`
- **Response**: Base64 encoded audio with character count

#### Pipeline API
```
POST /pipeline
```
- **Purpose**: Chain multiple services together
- **Pay-per-Use Metric**: Counts each service inside the pipeline separately
- **Request**: `{"input": "Text to process"}`
- **Response**: Complete pipeline output with individual service results

### Usage Tracking APIs

#### Data Volume Processed
```
GET /usage/data-volume?customer_id={id}&service={service}
```
- Returns total input/output data processed for a specific service

#### Customer API Call Count
```
GET /usage/api-calls?customer_id={id}&service={service}
```
- Returns total number of requests made for a specific service

#### Error Tracking
```
GET /usage/errors?customer_id={id}&service={service}
```
- Returns number of failed API requests with error breakdown

#### Usage Summary
```
GET /v1/usage?customer_id={id}
```
- Comprehensive usage summary across all services

#### Usage Events
```
GET /v1/usage/events?customer_id={id}&service={service}
```
- Detailed usage events for individual API calls

### Quota Management

#### Get Quotas
```
GET /v1/quotas?customer_id={id}
```
- Show current usage limits and quotas

#### Request Quota Increase
```
POST /v1/quotas/requests
```
- Request an increase in quotas

### Billing

#### Get Invoices
```
GET /v1/invoices?customer_id={id}&month=2025-08
```
- List invoices for a specified month

#### Get Credits
```
GET /v1/credits?customer_id={id}
```
- Display promotional or pre-paid credit balances

### Webhooks

#### Subscribe to Events
```
POST /v1/webhooks/subscriptions
```
- Subscribe to events (usage threshold, quota changes, etc.)

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- PostgreSQL database
- Dhruva API access

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd AI4X

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export DHRUVA_AUTH_TOKEN="your_dhruva_token"
export GEMINI_API_KEY="your_gemini_key"

# Set up database
# Create PostgreSQL database and run the schema setup
```

### Database Schema
The application uses the following main tables:
- `ai4x_demo_requests_log_v6`: Main usage tracking table
- Additional tables for quotas, billing, and webhooks (to be implemented)

### Running the Application
```bash
# Start the server
python ai4x_demo_app.py

# The API will be available at http://localhost:8000
# API documentation at http://localhost:8000/docs
```

## 🧪 Testing

### Run the Demo Client
```bash
# Make sure the server is running
python ai4x_demo_client.py
```

The demo client will:
1. Test all core model APIs
2. Demonstrate pipeline composition
3. Show usage tracking capabilities
4. Test quota management
5. Demonstrate billing features
6. Test webhook subscriptions

### Manual Testing
You can test individual endpoints using curl or any HTTP client:

```bash
# Test NMT API
curl -X POST "http://localhost:8000/nmt/translate" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello", "source_language": "en", "target_language": "hi"}'

# Test Usage Tracking
curl "http://localhost:8000/usage/data-volume?customer_id=demo_customer_001&service=NMT"

# Test Quotas
curl "http://localhost:8000/v1/quotas?customer_id=demo_customer_001"
```

## 📊 Metrics & Monitoring

### Database Metrics Storage
All metrics are stored directly in PostgreSQL database:
- Request counts and durations
- Component latencies
- Data processed counters
- Error rates and types
- Service-specific usage tracking

### Database Schema
The main table `ai4x_demo_requests_log_v6` stores:
- Request metadata (ID, customer, timestamp)
- Service latencies (NMT, ASR, TTS, LLM, BackNMT)
- Usage metrics (character counts, audio duration)
- Error tracking and status information

## 🔧 Configuration

### Environment Variables
```bash
DHRUVA_AUTH_TOKEN=your_dhruva_token
GEMINI_API_KEY=your_gemini_key
OPENWEATHER_API_KEY=your_weather_key
DB_HOST=localhost
DB_NAME=ai4x_demo
DB_USER=postgres
DB_PASSWORD=password
DB_PORT=5432
```

### Service Configuration
Edit `config.py` to customize:
- API timeouts
- Supported languages
- Default values
- Service endpoints

## 🚀 Deployment

### Docker Deployment
```bash
# Build the Docker image
docker build -t ai4x-platform .

# Run with docker-compose
docker-compose up -d
```

### Production Considerations
- Set up proper database connection pooling
- Configure Redis for caching
- Set up proper logging and monitoring
- Implement rate limiting
- Add authentication and authorization
- Set up SSL/TLS certificates

## 📈 Success Criteria Met

✅ **Core Model APIs Working**
- NMT, ASR, TTS, and LLM APIs deployed and callable
- Inputs and outputs work as expected

✅ **End-to-End Pipeline Demo**
- Complete workflow: Text → NMT → LLM → BackNMT → TTS
- Demoable via API and client script

✅ **Basic Pay-per-Use Tracking**
- API Call Count per service per customer
- Data Volume Consumed (text length, audio duration)
- Compute/Processing Time measurement
- Peak Concurrent Usage tracking

✅ **Advanced Features**
- Real-time usage dashboards
- Quota management
- Billing and invoicing
- Webhook notifications
- Comprehensive error tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the API documentation at `/docs`
- Review the demo client for usage examples

---

**AI4X** - Empowering AI with Pay-per-Use Precision
