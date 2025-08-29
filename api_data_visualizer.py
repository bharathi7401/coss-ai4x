#!/usr/bin/env python3
"""
API Aggregate Data Visualizer
Fetches and displays aggregate data from the AI4X API
"""

import requests
import json
from datetime import datetime

def fetch_customer_aggregates(customer_name):
    """Fetch aggregate data for a specific customer"""
    try:
        response = requests.get(f"http://localhost:8000/customer_aggregates?customerName={customer_name}")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching data for {customer_name}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def display_aggregate_data(customer_data):
    """Display aggregate data in a formatted way"""
    if not customer_data or not customer_data.get('aggregates'):
        print(f"No data available for {customer_data.get('customerName', 'unknown')}")
        return
    
    customer_name = customer_data['customerName']
    aggregates = customer_data['aggregates']
    
    print(f"\n{'=' * 60}")
    print(f"  📊 CUSTOMER {customer_name.upper()} AGGREGATE METRICS")
    print(f"{'=' * 60}")
    
    total_apps = len(aggregates)
    print(f"Total Applications: {total_apps}")
    
    # Calculate overall averages
    all_pipeline_latencies = [app['avg_overallPipelineLatency'] for app in aggregates if app['avg_overallPipelineLatency'] > 0]
    all_llm_latencies = [app['avg_llmLatency'] for app in aggregates if app['avg_llmLatency'] > 0]
    all_tts_latencies = [app['avg_ttsLatency'] for app in aggregates if app['avg_ttsLatency'] > 0]
    
    if all_pipeline_latencies:
        avg_pipeline = sum(all_pipeline_latencies) / len(all_pipeline_latencies)
        print(f"Overall Average Pipeline Latency: {avg_pipeline:.2f}ms")
    
    print(f"\n{'─' * 60}")
    print(f"  📱 APPLICATION BREAKDOWN")
    print(f"{'─' * 60}")
    
    for i, app in enumerate(aggregates, 1):
        app_name = app['customerApp']
        print(f"\n{i}. {app_name}")
        print(f"   ⏱️  Latencies (ms):")
        print(f"      • Language Detection: {app['avg_langdetectionLatency']:.2f}")
        print(f"      • NMT Translation:    {app['avg_nmtLatency']:.2f}")
        print(f"      • LLM Processing:     {app['avg_llmLatency']:.2f}")
        print(f"      • TTS Generation:     {app['avg_ttsLatency']:.2f}")
        print(f"      • Overall Pipeline:   {app['avg_overallPipelineLatency']:.2f}")
        
        print(f"   📈 Usage Statistics:")
        print(f"      • NMT Usage:          {app['avg_nmtUsage']:.2f}")
        print(f"      • LLM Usage:          {app['avg_llmUsage']:.2f}")
        print(f"      • TTS Usage:          {app['avg_ttsUsage']:.2f}")

def create_api_data_dashboard_panel():
    """Create a comprehensive dashboard panel with API data"""
    
    print(f"\n{'🚀' * 20}")
    print(f"  AI4X API AGGREGATE DATA REPORT")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'🚀' * 20}")
    
    # Fetch data for both customers
    customers = ['cust1', 'cust2']
    all_data = {}
    
    for customer in customers:
        print(f"\n📡 Fetching data for {customer}...")
        data = fetch_customer_aggregates(customer)
        if data:
            all_data[customer] = data
            display_aggregate_data(data)
        else:
            print(f"❌ No data available for {customer}")
    
    # Create comparison summary
    print(f"\n{'=' * 60}")
    print(f"  🔄 CUSTOMER COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    
    if len(all_data) >= 2:
        cust1_data = all_data.get('cust1', {}).get('aggregates', [])
        cust2_data = all_data.get('cust2', {}).get('aggregates', [])
        
        print(f"Customer 1 Applications: {len(cust1_data)}")
        print(f"Customer 2 Applications: {len(cust2_data)}")
        
        # Average pipeline latency comparison
        if cust1_data and cust2_data:
            cust1_avg = sum(app['avg_overallPipelineLatency'] for app in cust1_data if app['avg_overallPipelineLatency'] > 0) / len([app for app in cust1_data if app['avg_overallPipelineLatency'] > 0])
            cust2_avg = sum(app['avg_overallPipelineLatency'] for app in cust2_data if app['avg_overallPipelineLatency'] > 0) / len([app for app in cust2_data if app['avg_overallPipelineLatency'] > 0])
            
            print(f"\nAverage Pipeline Latency Comparison:")
            print(f"  • Customer 1: {cust1_avg:.2f}ms")
            print(f"  • Customer 2: {cust2_avg:.2f}ms")
            
            if cust1_avg > cust2_avg:
                diff = ((cust1_avg - cust2_avg) / cust2_avg) * 100
                print(f"  • Customer 1 is {diff:.1f}% slower than Customer 2")
            else:
                diff = ((cust2_avg - cust1_avg) / cust1_avg) * 100
                print(f"  • Customer 2 is {diff:.1f}% slower than Customer 1")
    
    print(f"\n{'📈' * 20}")
    print(f"  API ENDPOINTS FOR GRAFANA INTEGRATION")
    print(f"{'📈' * 20}")
    print(f"Customer 1: http://localhost:8000/customer_aggregates?customerName=cust1")
    print(f"Customer 2: http://localhost:8000/customer_aggregates?customerName=cust2")
    print(f"All Customers: http://localhost:8000/customers")
    print(f"API Docs: http://localhost:8000/docs")

if __name__ == "__main__":
    create_api_data_dashboard_panel()
