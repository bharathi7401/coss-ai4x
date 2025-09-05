# AI4X Pipeline Request Sender

This directory contains files for automatically sending sample requests to the AI4X pipeline endpoint.

## Files

### 1. `sample_requests.json`
Contains sample input data with Tamil and Hindi text for both customers:
- **cust1**: Voice assistant requests (NMT + LLM + TTS pipeline)
- **cust2**: Chat support requests (NMT + LLM pipeline)

Each request includes:
- `customerName`: "cust1" or "cust2"
- `customerAppName`: Application name
- `input.text`: Sample text in Tamil or Hindi
- `input.language`: Language code ("ta" for Tamil, "hi" for Hindi)

### 2. `request_sender.py`
Main script that sends requests to the `/pipeline` endpoint with the following features:
- Loads sample data from `sample_requests.json`
- Sends requests continuously with random intervals
- Supports both customers or filters by specific customer
- Comprehensive logging and error handling
- Command-line arguments for customization

## Usage

### Automatic (with Docker Compose)
When you run `docker compose up`, the request sender will automatically start and begin sending requests after a 15-second startup delay.

### Manual Usage
```bash
# Basic usage - continuous requests with default settings
python request_sender.py

# Custom interval and maximum requests
python request_sender.py --interval-min 2 --interval-max 6 --max-requests 50

# Send only cust1 requests
python request_sender.py --customer cust1

# Send single batch instead of continuous
python request_sender.py --mode single

# Custom application URL
python request_sender.py --url http://localhost:8100
```

### Command Line Arguments
- `--url`: Base URL of the AI4X application (default: http://ai4x-app-new:8000)
- `--sample-file`: Path to sample requests JSON file (default: sample_requests.json)
- `--mode`: continuous or single batch (default: continuous)
- `--interval-min`: Minimum interval between requests in seconds (default: 2.0)
- `--interval-max`: Maximum interval between requests in seconds (default: 5.0)
- `--max-requests`: Maximum number of requests to send (default: infinite)
- `--customer`: Filter by customer - cust1, cust2, or both (default: both)
- `--wait-startup`: Wait time before starting requests in seconds (default: 10)

## Sample Data

The sample data includes realistic scenarios:

### Tamil Examples (cust1 - Voice Assistant)
- Weather inquiries
- Book purchasing requests
- Help requests
- Restaurant bookings
- Doctor appointments

### Hindi Examples (cust1 - Voice Assistant)
- Weather inquiries
- Book purchasing requests
- Help requests
- Restaurant bookings
- Doctor appointments

### Tamil Examples (cust2 - Chat Support)
- Service issues
- Refund requests
- Product information
- Password recovery
- Customer care contact

### Hindi Examples (cust2 - Chat Support)
- Service issues
- Refund requests
- Product information
- Password recovery
- Customer care contact

## Docker Integration

The request sender is integrated into the Docker Compose setup:
- Waits 15 seconds for the main application to start
- Sends requests every 3-8 seconds (random interval)
- Automatically restarts if it fails
- Logs all activity to both console and `request_sender.log`

## Monitoring

The script provides detailed logging:
- Request success/failure status
- Response times and status codes
- Periodic statistics (every 10 requests)
- Error details for failed requests
- Final statistics on exit

All logs are written to both the console and `request_sender.log` file.
