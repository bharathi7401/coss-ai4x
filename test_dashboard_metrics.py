#!/usr/bin/env python3
"""
Test script to verify dashboard metrics are working correctly
"""
import requests
import json
import time
import random
from request_sender import RequestSender

def test_dashboard_metrics():
    """Test that all dashboard metrics are being generated correctly"""
    print("Testing dashboard metrics...")
    
    # Initialize request sender
    sender = RequestSender(base_url="http://localhost:8000", sample_file="sample_requests.json")
    
    # Send a few requests to generate metrics
    print("Sending test requests...")
    
    # Send pipeline requests
    cust1_requests = sender.sample_data.get('cust1_requests', [])
    cust2_requests = sender.sample_data.get('cust2_requests', [])
    
    if cust1_requests:
        print("Sending cust1 pipeline request...")
        sender.send_request(random.choice(cust1_requests))
    
    if cust2_requests:
        print("Sending cust2 pipeline request...")
        sender.send_request(random.choice(cust2_requests))
    
    # Send individual service requests
    individual_requests = sender.sample_data.get('individual_service_requests', {})
    
    if individual_requests.get('nmt_requests'):
        print("Sending NMT request...")
        sender.send_nmt_request(random.choice(individual_requests['nmt_requests']))
    
    if individual_requests.get('tts_requests'):
        print("Sending TTS request...")
        sender.send_tts_request(random.choice(individual_requests['tts_requests']))
    
    if individual_requests.get('llm_requests'):
        print("Sending LLM request...")
        sender.send_llm_request(random.choice(individual_requests['llm_requests']))
    
    # Wait a moment for metrics to be processed
    time.sleep(2)
    
    # Check metrics endpoint
    print("\nChecking metrics endpoint...")
    try:
        response = requests.get("http://localhost:8000/metrics")
        if response.status_code == 200:
            metrics_text = response.text
            
            # Check for key metrics used in dashboard
            dashboard_metrics = [
                "ai4x_requests_total",
                "ai4x_request_duration_seconds",
                "ai4x_errors_total", 
                "ai4x_throughput_requests_per_second",
                "ai4x_qos_availability_percent",
                "ai4x_service_requests_total",
                "ai4x_nmt_characters_translated_total",
                "ai4x_tts_characters_synthesized_total",
                "ai4x_llm_tokens_processed_total"
            ]
            
            found_metrics = []
            missing_metrics = []
            
            for metric in dashboard_metrics:
                if metric in metrics_text:
                    found_metrics.append(metric)
                else:
                    missing_metrics.append(metric)
            
            print(f"\n✅ Found metrics: {len(found_metrics)}")
            for metric in found_metrics:
                print(f"  - {metric}")
            
            if missing_metrics:
                print(f"\n❌ Missing metrics: {len(missing_metrics)}")
                for metric in missing_metrics:
                    print(f"  - {metric}")
            else:
                print("\n🎉 All dashboard metrics are available!")
            
            # Show sample metrics
            print("\nSample metrics:")
            lines = metrics_text.split('\n')
            for line in lines[:20]:  # Show first 20 lines
                if line and not line.startswith('#'):
                    print(f"  {line}")
            
            return len(missing_metrics) == 0
            
        else:
            print(f"❌ Metrics endpoint returned status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error checking metrics: {e}")
        return False

if __name__ == "__main__":
    print("Dashboard Metrics Test")
    print("=" * 50)
    
    success = test_dashboard_metrics()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ Dashboard metrics test PASSED")
        print("The dashboard should work correctly with your metrics!")
    else:
        print("❌ Dashboard metrics test FAILED")
        print("Some metrics may be missing or the application may not be running.")
