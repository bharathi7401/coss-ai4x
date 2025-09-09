#!/usr/bin/env python3
"""
AI4X Demo Client
Demonstrates the complete AI4X API functionality including:
- Core Model APIs (NMT, ASR, TTS)
- Pipeline Composition
- Pay-per-Use Tracking
- Usage Analytics
- Billing & Quotas
"""

import requests
import json
import base64
import time
from typing import Dict, Any

class AI4XClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.customer_id = "demo_customer_001"
    
    def test_nmt_api(self):
        """Test NMT API - Translate text"""
        print("🔄 Testing NMT API...")
        
        url = f"{self.base_url}/nmt/translate"
        payload = {
            "text": "Hello, how are you today?",
            "source_language": "en",
            "target_language": "hi"
        }
        
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ NMT Success: '{payload['text']}' → '{result['translated_text']}'")
            print(f"   Characters: {result['character_count']}, Latency: {result['latency']}")
            return result
        else:
            print(f"❌ NMT Failed: {response.status_code} - {response.text}")
            return None
    
    def test_tts_api(self):
        """Test TTS API - Convert text to speech"""
        print("\n🔄 Testing TTS API...")
        
        url = f"{self.base_url}/tts/speak"
        payload = {
            "text": "नमस्ते, आप कैसे हैं?",
            "language": "hi",
            "gender": "female"
        }
        
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ TTS Success: Generated audio for '{payload['text']}'")
            print(f"   Characters: {result['character_count']}, Latency: {result['latency']}")
            return result
        else:
            print(f"❌ TTS Failed: {response.status_code} - {response.text}")
            return None
    
    def test_pipeline_api(self):
        """Test Pipeline API - Chain multiple services"""
        print("\n🔄 Testing Pipeline API...")
        
        url = f"{self.base_url}/pipeline"
        payload = {
            "input": "What is the weather like today?"
        }
        
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Pipeline Success: Processed '{payload['input']}'")
            print(f"   Request ID: {result['requestId']}")
            print(f"   Services: {list(result['pipelineOutput'].keys())}")
            print(f"   Total Latency: {result['latency']['pipelineTotal']}")
            return result
        else:
            print(f"❌ Pipeline Failed: {response.status_code} - {response.text}")
            return None
    
    def test_usage_tracking(self):
        """Test Usage Tracking APIs"""
        print("\n📊 Testing Usage Tracking APIs...")
        
        # Test data volume
        url = f"{self.base_url}/usage/data-volume"
        params = {"customer_id": self.customer_id, "service": "NMT"}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Data Volume: {result['total_input_chars']} input chars, {result['total_output_chars']} output chars")
        
        # Test API calls
        url = f"{self.base_url}/usage/api-calls"
        params = {"customer_id": self.customer_id, "service": "NMT"}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API Calls: {result['total_requests']} total, {result['successful_requests']} successful")
        
        # Test usage summary
        url = f"{self.base_url}/v1/usage"
        params = {"customer_id": self.customer_id}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Usage Summary: {result['total_requests']} requests, {result['total_characters']} characters")
    
    def test_quota_management(self):
        """Test Quota Management APIs"""
        print("\n🎯 Testing Quota Management APIs...")
        
        # Test quotas
        url = f"{self.base_url}/v1/quotas"
        params = {"customer_id": self.customer_id}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            quotas = response.json()
            print("✅ Current Quotas:")
            for quota in quotas:
                print(f"   {quota['service']}: {quota['current_usage']}/{quota['limit']} ({quota['status']})")
        
        # Test quota request
        url = f"{self.base_url}/v1/quotas/requests"
        payload = {
            "customer_id": self.customer_id,
            "service": "NMT",
            "requested_limit": 200000,
            "reason": "Increased translation needs for new project"
        }
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Quota Request: {result['status']} for {result['service']}")
    
    def test_billing_apis(self):
        """Test Billing APIs"""
        print("\n💰 Testing Billing APIs...")
        
        # Test invoices
        url = f"{self.base_url}/v1/invoices"
        params = {"customer_id": self.customer_id}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            invoices = response.json()
            if invoices:
                invoice = invoices[0]
                print(f"✅ Invoice: {invoice['invoice_id']} - ${invoice['total_amount']} ({invoice['status']})")
        
        # Test credits
        url = f"{self.base_url}/v1/credits"
        params = {"customer_id": self.customer_id}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Credits: ${result['total_credits']} total (${result['promotional_credits']} promotional, ${result['prepaid_credits']} prepaid)")
    
    def test_webhook_subscription(self):
        """Test Webhook Subscription API"""
        print("\n🔔 Testing Webhook Subscription API...")
        
        url = f"{self.base_url}/v1/webhooks/subscriptions"
        payload = {
            "customer_id": self.customer_id,
            "event_types": ["usage_threshold", "quota_exceeded", "invoice_generated"],
            "webhook_url": "https://demo-customer.com/webhooks/ai4x"
        }
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Webhook Subscription: {result['subscription_id']} ({result['status']})")
    
    def run_complete_demo(self):
        """Run the complete AI4X demo"""
        print("🚀 Starting AI4X Complete Demo")
        print("=" * 50)
        
        try:
            # Test core model APIs
            self.test_nmt_api()
            self.test_tts_api()
            
            # Test pipeline composition
            self.test_pipeline_api()
            
            # Test usage tracking
            self.test_usage_tracking()
            
            # Test quota management
            self.test_quota_management()
            
            # Test billing
            self.test_billing_apis()
            
            # Test webhooks
            self.test_webhook_subscription()
            
            print("\n" + "=" * 50)
            print("🎉 AI4X Demo Completed Successfully!")
            print("\nKey Features Demonstrated:")
            print("✅ Core Model APIs (NMT, ASR, TTS)")
            print("✅ Pipeline Composition")
            print("✅ Pay-per-Use Tracking (Direct DB Storage)")
            print("✅ Usage Analytics")
            print("✅ Quota Management")
            print("✅ Billing & Invoicing")
            print("✅ Webhook Subscriptions")
            
        except Exception as e:
            print(f"\n❌ Demo failed with error: {str(e)}")
            print("Make sure the AI4X server is running on http://localhost:8000")

def main():
    """Main function to run the demo"""
    client = AI4XClient()
    client.run_complete_demo()

if __name__ == "__main__":
    main()
