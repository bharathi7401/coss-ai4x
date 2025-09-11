#!/usr/bin/env python3
"""
Test script to verify CPU and memory usage metrics are working correctly
"""
import requests
import time
import json

def test_system_metrics():
    """Test that CPU and memory usage metrics are being captured"""
    print("Testing system metrics collection...")
    
    base_url = "http://localhost:8000"
    
    print("\n1. Testing manual system metrics collection...")
    try:
        response = requests.post(f"{base_url}/debug/collect-system-metrics")
        if response.status_code == 200:
            result = response.json()
            print(f"  ✅ Manual collection: {result}")
        else:
            print(f"  ❌ Manual collection failed: {response.status_code}")
    except Exception as e:
        print(f"  ❌ Error calling manual collection: {e}")
    
    print("\n2. Checking current metrics...")
    check_system_metrics()
    
    print("\n3. Waiting for automatic collection (10 seconds)...")
    time.sleep(10)
    
    print("\n4. Checking metrics after automatic collection...")
    check_system_metrics()
    
    print("\n5. Testing multiple manual collections...")
    for i in range(3):
        print(f"  Collection {i+1}:")
        try:
            response = requests.post(f"{base_url}/debug/collect-system-metrics")
            if response.status_code == 200:
                result = response.json()
                print(f"    ✅ {result}")
            else:
                print(f"    ❌ Failed: {response.status_code}")
        except Exception as e:
            print(f"    ❌ Error: {e}")
        time.sleep(2)
    
    print("\n6. Final metrics check...")
    check_system_metrics()

def check_system_metrics():
    """Check current system metrics values"""
    try:
        response = requests.get("http://localhost:8000/metrics")
        if response.status_code == 200:
            metrics_text = response.text
            
            print("  📊 Current System Metrics:")
            
            # Extract CPU usage
            cpu_usage = extract_metric_value(metrics_text, 'ai4x_cpu_usage_percent')
            if cpu_usage is not None:
                print(f"    CPU Usage: {cpu_usage:.2f}%")
            else:
                print("    CPU Usage: Not found")
            
            # Extract memory usage
            memory_usage = extract_metric_value(metrics_text, 'ai4x_memory_usage_percent')
            if memory_usage is not None:
                print(f"    Memory Usage: {memory_usage:.2f}%")
            else:
                print("    Memory Usage: Not found")
            
            # Extract raw CPU usage
            cpu_raw = extract_metric_value(metrics_text, 'ai4x_cpu_usage')
            if cpu_raw is not None:
                print(f"    CPU Raw: {cpu_raw:.2f}")
            
            # Extract raw memory usage
            memory_raw = extract_metric_value(metrics_text, 'ai4x_memory_usage')
            if memory_raw is not None:
                print(f"    Memory Raw: {memory_raw / (1024**3):.2f} GB")
            
            # Check if metrics are being updated
            if cpu_usage == 0.0 and memory_usage == 0.0:
                print("    ⚠️  WARNING: Both CPU and memory usage are 0% - this might indicate an issue")
            elif cpu_usage is not None and memory_usage is not None:
                print("    ✅ System metrics are being captured")
            
        else:
            print(f"  ❌ Metrics endpoint returned status {response.status_code}")
            
    except Exception as e:
        print(f"  ❌ Error checking system metrics: {e}")

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

def test_psutil_directly():
    """Test psutil directly to see if it's working"""
    print("\n🔧 Testing psutil directly...")
    try:
        import psutil
        
        # Test CPU
        cpu_percent = psutil.cpu_percent(interval=None)
        print(f"  CPU percent (non-blocking): {cpu_percent}%")
        
        cpu_percent_blocking = psutil.cpu_percent(interval=1)
        print(f"  CPU percent (blocking): {cpu_percent_blocking}%")
        
        # Test Memory
        mem = psutil.virtual_memory()
        print(f"  Memory percent: {mem.percent}%")
        print(f"  Memory used: {mem.used / (1024**3):.2f} GB")
        print(f"  Memory total: {mem.total / (1024**3):.2f} GB")
        
        return True
    except Exception as e:
        print(f"  ❌ psutil test failed: {e}")
        return False

if __name__ == "__main__":
    print("System Metrics Collection Test")
    print("=" * 50)
    
    # First test psutil directly
    if test_psutil_directly():
        print("\n" + "=" * 50)
        test_system_metrics()
    else:
        print("\n❌ psutil is not working properly. Please check your Python environment.")
    
    print("\n" + "=" * 50)
    print("✅ System metrics test completed!")
    print("\nTroubleshooting tips:")
    print("- If CPU/Memory show 0%, check if the background thread is running")
    print("- Look for DEBUG messages in the application logs")
    print("- Try the manual collection endpoint: POST /debug/collect-system-metrics")
    print("- Check if psutil is installed: pip install psutil")
