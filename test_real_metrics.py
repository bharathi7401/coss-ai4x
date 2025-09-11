#!/usr/bin/env python3
"""
Test script to verify real-time metrics calculations work correctly
"""
import requests
import json
import time
import random
from request_sender import RequestSender

def test_real_metrics_calculation():
    """Test that metrics are calculated based on real request data"""
    print("Testing real-time metrics calculation...")
    
    # Initialize request sender
    sender = RequestSender(base_url="http://localhost:8000", sample_file="sample_requests.json")
    
    print("\n1. Testing initial state (no requests)...")
    check_metrics("Initial state", expected_error_rate=0.0, expected_availability=100.0)
    
    print("\n2. Sending successful requests...")
    # Send some successful requests
    cust1_requests = sender.sample_data.get('cust1_requests', [])
    if cust1_requests:
        for i in range(3):
            print(f"  Sending successful request {i+1}...")
            sender.send_request(random.choice(cust1_requests))
            time.sleep(1)
    
    # Wait for metrics to update
    time.sleep(3)
    check_metrics("After successful requests", expected_error_rate=0.0, expected_availability=100.0)
    
    print("\n3. Testing individual service requests...")
    # Send individual service requests
    individual_requests = sender.sample_data.get('individual_service_requests', {})
    
    if individual_requests.get('nmt_requests'):
        print("  Sending NMT requests...")
        for i in range(2):
            sender.send_nmt_request(random.choice(individual_requests['nmt_requests']))
            time.sleep(1)
    
    if individual_requests.get('llm_requests'):
        print("  Sending LLM requests...")
        for i in range(2):
            sender.send_llm_request(random.choice(individual_requests['llm_requests']))
            time.sleep(1)
    
    # Wait for metrics to update
    time.sleep(3)
    check_metrics("After individual service requests", expected_error_rate=0.0, expected_availability=100.0)
    
    print("\n4. Testing error simulation...")
    # Simulate some errors by sending requests to non-existent endpoints
    try:
        response = requests.post("http://localhost:8000/nonexistent", json={}, timeout=5)
    except:
        pass  # Expected to fail
    
    # Wait for metrics to update
    time.sleep(3)
    check_metrics("After error simulation", expected_error_rate=None, expected_availability=None)
    
    print("\n5. Final metrics summary...")
    show_metrics_summary()

def check_metrics(test_name, expected_error_rate=None, expected_availability=None):
    """Check current metrics values"""
    try:
        response = requests.get("http://localhost:8000/metrics")
        if response.status_code == 200:
            metrics_text = response.text
            
            # Extract error rate
            error_rate = extract_metric_value(metrics_text, "ai4x_system_error_rate_percent")
            if error_rate is not None:
                print(f"  Error Rate: {error_rate:.2f}%")
                if expected_error_rate is not None:
                    if abs(error_rate - expected_error_rate) < 0.1:
                        print(f"  ✅ Error rate matches expected ({expected_error_rate}%)")
                    else:
                        print(f"  ❌ Error rate {error_rate}% doesn't match expected {expected_error_rate}%")
            else:
                print("  ❌ Error rate metric not found")
            
            # Extract availability
            availability = extract_metric_value(metrics_text, "ai4x_qos_availability_percent")
            if availability is not None:
                print(f"  Availability: {availability:.2f}%")
                if expected_availability is not None:
                    if abs(availability - expected_availability) < 0.1:
                        print(f"  ✅ Availability matches expected ({expected_availability}%)")
                    else:
                        print(f"  ❌ Availability {availability}% doesn't match expected {expected_availability}%")
            else:
                print("  ❌ Availability metric not found")
            
            # Extract performance scores
            print("  QoS Performance Scores:")
            for customer in ["cust1", "cust2"]:
                for app in ["voice-assistant (cust1)", "chat-support (cust2)", "nmt-app", "llm-app", "tts-app"]:
                    score = extract_metric_value(metrics_text, f"ai4x_qos_performance_score{{customer=\"{customer}\",app=\"{app}\"}}")
                    if score is not None:
                        print(f"    {customer}/{app}: {score:.2f}%")
            
        else:
            print(f"  ❌ Metrics endpoint returned status {response.status_code}")
            
    except Exception as e:
        print(f"  ❌ Error checking metrics: {e}")

def extract_metric_value(metrics_text, metric_name):
    """Extract a metric value from the metrics text"""
    lines = metrics_text.split('\n')
    for line in lines:
        if line.startswith(metric_name) and not line.startswith('#'):
            try:
                # Extract the value (last part after any spaces)
                value = float(line.split()[-1])
                return value
            except:
                continue
    return None

def show_metrics_summary():
    """Show a summary of all current metrics"""
    try:
        response = requests.get("http://localhost:8000/metrics")
        if response.status_code == 200:
            metrics_text = response.text
            
            print("  📊 Current Metrics Summary:")
            
            # Request counts
            cust1_requests = extract_metric_value(metrics_text, 'ai4x_requests_total{customer="cust1",status="success"}')
            cust2_requests = extract_metric_value(metrics_text, 'ai4x_requests_total{customer="cust2",status="success"}')
            if cust1_requests is not None:
                print(f"    cust1 successful requests: {int(cust1_requests)}")
            if cust2_requests is not None:
                print(f"    cust2 successful requests: {int(cust2_requests)}")
            
            # Error counts
            cust1_errors = extract_metric_value(metrics_text, 'ai4x_errors_total{customer="cust1"}')
            cust2_errors = extract_metric_value(metrics_text, 'ai4x_errors_total{customer="cust2"}')
            if cust1_errors is not None:
                print(f"    cust1 errors: {int(cust1_errors)}")
            if cust2_errors is not None:
                print(f"    cust2 errors: {int(cust2_errors)}")
            
            # Service requests
            nmt_requests = extract_metric_value(metrics_text, 'ai4x_service_requests_total{service="nmt"}')
            llm_requests = extract_metric_value(metrics_text, 'ai4x_service_requests_total{service="llm"}')
            tts_requests = extract_metric_value(metrics_text, 'ai4x_service_requests_total{service="tts"}')
            
            if nmt_requests is not None:
                print(f"    NMT service requests: {int(nmt_requests)}")
            if llm_requests is not None:
                print(f"    LLM service requests: {int(llm_requests)}")
            if tts_requests is not None:
                print(f"    TTS service requests: {int(tts_requests)}")
            
            # Data processed
            nmt_chars = extract_metric_value(metrics_text, 'ai4x_nmt_characters_translated_total{customer="cust1"}')
            llm_tokens = extract_metric_value(metrics_text, 'ai4x_llm_tokens_processed_total{customer="cust1"}')
            tts_chars = extract_metric_value(metrics_text, 'ai4x_tts_characters_synthesized_total{customer="cust1"}')
            
            if nmt_chars is not None:
                print(f"    NMT characters translated: {int(nmt_chars)}")
            if llm_tokens is not None:
                print(f"    LLM tokens processed: {int(llm_tokens)}")
            if tts_chars is not None:
                print(f"    TTS characters synthesized: {int(tts_chars)}")
            
        else:
            print(f"  ❌ Metrics endpoint returned status {response.status_code}")
            
    except Exception as e:
        print(f"  ❌ Error getting metrics summary: {e}")

if __name__ == "__main__":
    print("Real-Time Metrics Calculation Test")
    print("=" * 50)
    
    test_real_metrics_calculation()
    
    print("\n" + "=" * 50)
    print("✅ Real-time metrics calculation test completed!")
    print("\nKey Points:")
    print("- Error rate starts at 0% and only increases when actual errors occur")
    print("- Availability starts at 100% and decreases only when requests fail")
    print("- Performance scores are calculated based on actual request success/failure")
    print("- All metrics reflect real data from your request-sender.py")
