#!/usr/bin/env python3
"""
AI4X Pipeline Request Sender

This script sends sample requests to the AI4X pipeline endpoint.
It loads sample data from sample_requests.json and sends requests
to both cust1 and cust2 pipelines with configurable intervals.
"""

import json
import requests
import time
import random
import logging
import sys
from datetime import datetime
from typing import List, Dict, Any
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('request_sender.log')
    ]
)
logger = logging.getLogger(__name__)

class RequestSender:
    def __init__(self, base_url: str = "http://ai4x-app-new:8000", sample_file: str = "sample_requests.json"):
        """
        Initialize the request sender.
        
        Args:
            base_url: Base URL of the AI4X application
            sample_file: Path to the sample requests JSON file
        """
        self.base_url = base_url
        self.sample_file = sample_file
        self.pipeline_endpoint = f"{base_url}/pipeline"
        self.sample_data = self.load_sample_data()
        
    def load_sample_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load sample requests from JSON file."""
        try:
            with open(self.sample_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data.get('cust1_requests', []))} cust1 requests and {len(data.get('cust2_requests', []))} cust2 requests")
            return data
        except FileNotFoundError:
            logger.error(f"Sample data file {self.sample_file} not found!")
            sys.exit(1)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file: {e}")
            sys.exit(1)
    
    def send_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a single request to the pipeline endpoint.
        
        Args:
            request_data: Request payload
            
        Returns:
            Response data or error information
        """
        try:
            logger.info(f"Sending request for {request_data['customerName']} - {request_data['input']['text'][:50]}...")
            
            response = requests.post(
                self.pipeline_endpoint,
                json=request_data,
                timeout=30,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Request successful - Status: {result.get('status', 'unknown')}")
                return {
                    'success': True,
                    'status_code': response.status_code,
                    'response': result
                }
            else:
                logger.error(f"❌ Request failed with status {response.status_code}: {response.text}")
                return {
                    'success': False,
                    'status_code': response.status_code,
                    'error': response.text
                }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Request failed with exception: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def send_requests_continuously(self, interval_min: float = 2.0, interval_max: float = 5.0, 
                                 max_requests: int = None, customer_filter: str = None):
        """
        Send requests continuously with random intervals.
        
        Args:
            interval_min: Minimum interval between requests (seconds)
            interval_max: Maximum interval between requests (seconds)
            max_requests: Maximum number of requests to send (None for infinite)
            customer_filter: Filter by customer ('cust1' or 'cust2', None for both)
        """
        logger.info(f"Starting continuous request sending...")
        logger.info(f"Interval: {interval_min}-{interval_max} seconds")
        logger.info(f"Max requests: {max_requests if max_requests else 'infinite'}")
        logger.info(f"Customer filter: {customer_filter if customer_filter else 'all'}")
        
        request_count = 0
        success_count = 0
        error_count = 0
        
        try:
            while True:
                if max_requests and request_count >= max_requests:
                    logger.info(f"Reached maximum requests limit ({max_requests})")
                    break
                
                # Select requests based on customer filter
                if customer_filter == 'cust1':
                    available_requests = self.sample_data.get('cust1_requests', [])
                elif customer_filter == 'cust2':
                    available_requests = self.sample_data.get('cust2_requests', [])
                else:
                    # Mix both customers
                    cust1_requests = self.sample_data.get('cust1_requests', [])
                    cust2_requests = self.sample_data.get('cust2_requests', [])
                    available_requests = cust1_requests + cust2_requests
                
                if not available_requests:
                    logger.error("No sample requests available!")
                    break
                
                # Select a random request
                request_data = random.choice(available_requests)
                
                # Send the request
                result = self.send_request(request_data)
                
                request_count += 1
                if result['success']:
                    success_count += 1
                else:
                    error_count += 1
                
                # Log statistics
                if request_count % 10 == 0:
                    logger.info(f"📊 Statistics - Total: {request_count}, Success: {success_count}, Errors: {error_count}")
                
                # Wait before next request
                interval = random.uniform(interval_min, interval_max)
                logger.info(f"⏳ Waiting {interval:.1f} seconds before next request...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("🛑 Request sending stopped by user")
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
        finally:
            logger.info(f"📊 Final Statistics - Total: {request_count}, Success: {success_count}, Errors: {error_count}")
    
    def send_single_batch(self, customer: str = None):
        """
        Send one request from each customer (or specified customer).
        
        Args:
            customer: Customer to send request for ('cust1', 'cust2', or None for both)
        """
        logger.info("Sending single batch of requests...")
        
        if customer == 'cust1' or customer is None:
            cust1_requests = self.sample_data.get('cust1_requests', [])
            if cust1_requests:
                request_data = random.choice(cust1_requests)
                self.send_request(request_data)
        
        if customer == 'cust2' or customer is None:
            cust2_requests = self.sample_data.get('cust2_requests', [])
            if cust2_requests:
                request_data = random.choice(cust2_requests)
                self.send_request(request_data)

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='AI4X Pipeline Request Sender')
    parser.add_argument('--url', default='http://ai4x-app-new:8000', 
                       help='Base URL of the AI4X application (default: http://ai4x-app-new:8000)')
    parser.add_argument('--sample-file', default='sample_requests.json',
                       help='Path to sample requests JSON file (default: sample_requests.json)')
    parser.add_argument('--mode', choices=['continuous', 'single'], default='continuous',
                       help='Mode: continuous or single batch (default: continuous)')
    parser.add_argument('--interval-min', type=float, default=2.0,
                       help='Minimum interval between requests in seconds (default: 2.0)')
    parser.add_argument('--interval-max', type=float, default=5.0,
                       help='Maximum interval between requests in seconds (default: 5.0)')
    parser.add_argument('--max-requests', type=int, default=None,
                       help='Maximum number of requests to send (default: infinite)')
    parser.add_argument('--customer', choices=['cust1', 'cust2'], default=None,
                       help='Filter by customer (default: both)')
    parser.add_argument('--wait-startup', type=int, default=10,
                       help='Wait time in seconds before starting requests (default: 10)')
    
    args = parser.parse_args()
    
    # Wait for application to start up
    logger.info(f"⏳ Waiting {args.wait_startup} seconds for application startup...")
    time.sleep(args.wait_startup)
    
    # Initialize request sender
    sender = RequestSender(base_url=args.url, sample_file=args.sample_file)
    
    # Test connection first
    logger.info("🔍 Testing connection to AI4X application...")
    try:
        health_response = requests.get(f"{args.url}/", timeout=5)
        if health_response.status_code == 200:
            logger.info("✅ Connection successful!")
        else:
            logger.warning(f"⚠️  Application responded with status {health_response.status_code}")
    except Exception as e:
        logger.error(f"❌ Cannot connect to application: {e}")
        logger.info("🔄 Continuing anyway - application might still be starting up...")
    
    # Send requests based on mode
    if args.mode == 'continuous':
        sender.send_requests_continuously(
            interval_min=args.interval_min,
            interval_max=args.interval_max,
            max_requests=args.max_requests,
            customer_filter=args.customer
        )
    else:
        sender.send_single_batch(customer=args.customer)

if __name__ == "__main__":
    main()
