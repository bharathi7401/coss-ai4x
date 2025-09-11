#!/usr/bin/env python3
"""
Test script to verify system uptime/availability metrics work correctly
"""
import requests
import json
import time
import random
from request_sender import RequestSender

def test_system_uptime_metrics():
    """Test that system uptime metrics are calculated based on actual failures"""
    print("Testing system uptime/availability metrics...")
    
    # Initialize request sender
    sender = RequestSender(base_url="http://localhost:8000", sample_file="sample_requests.json")
    
    print("\n1. Testing initial state (no failures)...")
    check_system_uptime("Initial state", expected_uptime=100.0)
    
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
    check_system_uptime("After successful requests", expected_uptime=100.0)
    
    print("\n3. Testing API errors...")
    # Simulate API errors by sending requests to non-existent endpoints
    for i in range(2):
        try:
            print(f"  Sending request to non-existent endpoint {i+1}...")
            response = requests.post("http://localhost:8000/nonexistent", json={}, timeout=5)
        except:
            pass  # Expected to fail
    
    # Wait for metrics to update
    time.sleep(3)
    check_system_uptime("After API errors", expected_uptime=None)
    
    print("\n4. Testing service errors...")
    # Send requests that might cause service errors
    individual_requests = sender.sample_data.get('individual_service_requests', {})
    
    if individual_requests.get('nmt_requests'):
        print("  Sending NMT requests...")
        for i in range(2):
            sender.send_nmt_request(random.choice(individual_requests['nmt_requests']))
            time.sleep(1)
    
    # Wait for metrics to update
    time.sleep(3)
    check_system_uptime("After service requests", expected_uptime=None)
    
    print("\n5. Final system uptime summary...")
    show_system_uptime_summary()

def check_system_uptime(test_name, expected_uptime=None):
    """Check current system uptime values"""
    try:
        response = requests.get("http://localhost:8000/metrics")
        if response.status_code == 200:
            metrics_text = response.text
            
            print(f"  📊 {test_name}:")
            
            # Extract system uptime for different time windows
            uptime_1h = extract_metric_value(metrics_text, 'ai4x_system_uptime_percent{time_window="1h"}')
            uptime_24h = extract_metric_value(metrics_text, 'ai4x_system_uptime_percent{time_window="24h"}')
            uptime_7d = extract_metric_value(metrics_text, 'ai4x_system_uptime_percent{time_window="7d"}')
            
            if uptime_1h is not None:
                print(f"    System Uptime (1h): {uptime_1h:.2f}%")
                if expected_uptime is not None:
                    if abs(uptime_1h - expected_uptime) < 0.1:
                        print(f"    ✅ Uptime matches expected ({expected_uptime}%)")
                    else:
                        print(f"    ❌ Uptime {uptime_1h}% doesn't match expected {expected_uptime}%")
            
            if uptime_24h is not None:
                print(f"    System Uptime (24h): {uptime_24h:.2f}%")
            
            if uptime_7d is not None:
                print(f"    System Uptime (7d): {uptime_7d:.2f}%")
            
            # Extract availability failures
            api_failures = extract_metric_value(metrics_text, 'ai4x_system_availability_failures_total{failure_type="api_error",component="api"}')
            service_failures = extract_metric_value(metrics_text, 'ai4x_system_availability_failures_total{failure_type="service_error"}')
            
            if api_failures is not None:
                print(f"    API Failures: {int(api_failures)}")
            
            if service_failures is not None:
                print(f"    Service Failures: {int(service_failures)}")
            
            # Extract regular availability for comparison
            availability = extract_metric_value(metrics_text, 'ai4x_qos_availability_percent{time_window="1h"}')
            if availability is not None:
                print(f"    QoS Availability: {availability:.2f}%")
            
        else:
            print(f"  ❌ Metrics endpoint returned status {response.status_code}")
            
    except Exception as e:
        print(f"  ❌ Error checking system uptime: {e}")

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

def show_system_uptime_summary():
    """Show a summary of all system uptime metrics"""
    try:
        response = requests.get("http://localhost:8000/metrics")
        if response.status_code == 200:
            metrics_text = response.text
            
            print("  📊 System Uptime Metrics Summary:")
            
            # System uptime for different time windows
            uptime_1h = extract_metric_value(metrics_text, 'ai4x_system_uptime_percent{time_window="1h"}')
            uptime_24h = extract_metric_value(metrics_text, 'ai4x_system_uptime_percent{time_window="24h"}')
            uptime_7d = extract_metric_value(metrics_text, 'ai4x_system_uptime_percent{time_window="7d"}')
            
            print(f"    System Uptime (1h): {uptime_1h:.2f}%" if uptime_1h is not None else "    System Uptime (1h): N/A")
            print(f"    System Uptime (24h): {uptime_24h:.2f}%" if uptime_24h is not None else "    System Uptime (24h): N/A")
            print(f"    System Uptime (7d): {uptime_7d:.2f}%" if uptime_7d is not None else "    System Uptime (7d): N/A")
            
            # Availability failures breakdown
            print("\n    Availability Failures:")
            api_failures = extract_metric_value(metrics_text, 'ai4x_system_availability_failures_total{failure_type="api_error",component="api"}')
            nmt_failures = extract_metric_value(metrics_text, 'ai4x_system_availability_failures_total{failure_type="service_error",component="nmt"}')
            llm_failures = extract_metric_value(metrics_text, 'ai4x_system_availability_failures_total{failure_type="service_error",component="llm"}')
            tts_failures = extract_metric_value(metrics_text, 'ai4x_system_availability_failures_total{failure_type="service_error",component="tts"}')
            
            print(f"      API Errors: {int(api_failures) if api_failures is not None else 0}")
            print(f"      NMT Service Errors: {int(nmt_failures) if nmt_failures is not None else 0}")
            print(f"      LLM Service Errors: {int(llm_failures) if llm_failures is not None else 0}")
            print(f"      TTS Service Errors: {int(tts_failures) if tts_failures is not None else 0}")
            
            # Total requests and errors for comparison
            total_requests = extract_metric_value(metrics_text, 'ai4x_requests_total{status="success"}')
            total_errors = extract_metric_value(metrics_text, 'ai4x_errors_total')
            
            if total_requests is not None and total_errors is not None:
                total_requests_int = int(total_requests)
                total_errors_int = int(total_errors)
                print(f"\n    Request Summary:")
                print(f"      Total Successful Requests: {total_requests_int}")
                print(f"      Total Errors: {total_errors_int}")
                if total_requests_int > 0:
                    error_rate = (total_errors_int / (total_requests_int + total_errors_int)) * 100
                    print(f"      Calculated Error Rate: {error_rate:.2f}%")
                    print(f"      Calculated Uptime: {100 - error_rate:.2f}%")
            
        else:
            print(f"  ❌ Metrics endpoint returned status {response.status_code}")
            
    except Exception as e:
        print(f"  ❌ Error getting system uptime summary: {e}")

if __name__ == "__main__":
    print("System Uptime/Availability Metrics Test")
    print("=" * 50)
    
    test_system_uptime_metrics()
    
    print("\n" + "=" * 50)
    print("✅ System uptime metrics test completed!")
    print("\nKey Points:")
    print("- System uptime starts at 100% and decreases when failures occur")
    print("- API errors and service errors are tracked separately")
    print("- Uptime is calculated as: 100% - (failures / total_requests) * 100%")
    print("- Different time windows (1h, 24h, 7d) are available for monitoring")
    print("- System uptime reflects actual availability failures, not just request errors")
