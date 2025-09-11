#!/usr/bin/env python3
"""
Simple script to verify system metrics are exposed via /metrics endpoint
"""
import requests
import time

def verify_metrics_endpoint():
    """Verify that system metrics are available via /metrics endpoint"""
    print("Verifying system metrics via /metrics endpoint...")
    
    try:
        response = requests.get("http://localhost:8000/metrics")
        if response.status_code == 200:
            metrics_text = response.text
            
            print("✅ /metrics endpoint is accessible")
            
            # Check for CPU metrics
            if "ai4x_cpu_usage_percent" in metrics_text:
                print("✅ CPU usage metric found")
                # Extract CPU value
                for line in metrics_text.split('\n'):
                    if line.startswith('ai4x_cpu_usage_percent') and not line.startswith('#'):
                        cpu_value = float(line.split()[-1])
                        print(f"   CPU Usage: {cpu_value}%")
                        break
            else:
                print("❌ CPU usage metric not found")
            
            # Check for memory metrics
            if "ai4x_memory_usage_percent" in metrics_text:
                print("✅ Memory usage metric found")
                # Extract memory value
                for line in metrics_text.split('\n'):
                    if line.startswith('ai4x_memory_usage_percent') and not line.startswith('#'):
                        mem_value = float(line.split()[-1])
                        print(f"   Memory Usage: {mem_value}%")
                        break
            else:
                print("❌ Memory usage metric not found")
            
            # Check for other system metrics
            system_metrics = [
                "ai4x_cpu_usage",
                "ai4x_memory_usage", 
                "ai4x_system_error_rate_percent",
                "ai4x_system_uptime_percent",
                "ai4x_qos_availability_percent"
            ]
            
            print("\n📊 System Metrics Status:")
            for metric in system_metrics:
                if metric in metrics_text:
                    print(f"   ✅ {metric}")
                else:
                    print(f"   ❌ {metric}")
            
            # Check if metrics are being updated (wait and check again)
            print("\n⏳ Waiting 10 seconds to check if metrics are updating...")
            time.sleep(10)
            
            response2 = requests.get("http://localhost:8000/metrics")
            if response2.status_code == 200:
                metrics_text2 = response2.text
                
                # Check if CPU value changed
                for line in metrics_text2.split('\n'):
                    if line.startswith('ai4x_cpu_usage_percent') and not line.startswith('#'):
                        cpu_value2 = float(line.split()[-1])
                        print(f"   CPU Usage (after 10s): {cpu_value2}%")
                        if cpu_value2 != cpu_value:
                            print("   ✅ CPU metrics are updating")
                        else:
                            print("   ⚠️  CPU metrics may not be updating")
                        break
                
                # Check if memory value changed
                for line in metrics_text2.split('\n'):
                    if line.startswith('ai4x_memory_usage_percent') and not line.startswith('#'):
                        mem_value2 = float(line.split()[-1])
                        print(f"   Memory Usage (after 10s): {mem_value2}%")
                        if mem_value2 != mem_value:
                            print("   ✅ Memory metrics are updating")
                        else:
                            print("   ⚠️  Memory metrics may not be updating")
                        break
            
        else:
            print(f"❌ /metrics endpoint returned status {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error accessing /metrics endpoint: {e}")

if __name__ == "__main__":
    print("System Metrics Verification")
    print("=" * 40)
    verify_metrics_endpoint()
    print("\n" + "=" * 40)
    print("✅ Verification completed!")
    print("\nNote: System metrics are collected automatically every 5 seconds")
    print("and exposed via the /metrics endpoint for Prometheus scraping.")
