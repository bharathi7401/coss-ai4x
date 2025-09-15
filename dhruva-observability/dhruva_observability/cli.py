"""
Command Line Interface for Dhruva Observability Plugin

Provides CLI tools for plugin management, configuration, and testing.
"""
import argparse
import json
import sys
from typing import Dict, Any
from .config import PluginConfig
from .metrics import MetricsCollector


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Dhruva Observability Plugin CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  dhruva-observability --status
  dhruva-observability --config
  dhruva-observability --test-metrics
  dhruva-observability --validate-config
        """
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="dhruva-observability 1.0.0"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show plugin status")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Show configuration")
    config_parser.add_argument(
        "--format",
        choices=["json", "yaml", "env"],
        default="json",
        help="Output format"
    )
    
    # Test metrics command
    test_parser = subparsers.add_parser("test-metrics", help="Test metrics collection")
    
    # Validate config command
    validate_parser = subparsers.add_parser("validate-config", help="Validate configuration")
    
    # Generate config command
    generate_parser = subparsers.add_parser("generate-config", help="Generate sample configuration")
    generate_parser.add_argument(
        "--output",
        help="Output file path"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "status":
            show_status()
        elif args.command == "config":
            show_config(args.format)
        elif args.command == "test-metrics":
            test_metrics()
        elif args.command == "validate-config":
            validate_config()
        elif args.command == "generate-config":
            generate_config(args.output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def show_status():
    """Show plugin status."""
    config = PluginConfig.from_env()
    
    print("Dhruva Observability Plugin Status")
    print("=" * 40)
    print(f"Enabled: {'✅ Yes' if config.enabled else '❌ No'}")
    print(f"Debug: {'✅ Yes' if config.debug else '❌ No'}")
    print(f"Customers: {', '.join(config.customers)}")
    print(f"Apps: {', '.join(config.apps)}")
    print(f"Metrics Path: {config.metrics_path}")
    print(f"Health Path: {config.health_path}")
    print(f"System Metrics: {'✅ Yes' if config.collect_system_metrics else '❌ No'}")
    print(f"GPU Metrics: {'✅ Yes' if config.collect_gpu_metrics else '❌ No'}")
    print(f"DB Metrics: {'✅ Yes' if config.collect_db_metrics else '❌ No'}")


def show_config(format_type: str):
    """Show configuration in specified format."""
    config = PluginConfig.from_env()
    
    if format_type == "json":
        print(json.dumps(config.to_dict(), indent=2))
    elif format_type == "yaml":
        try:
            import yaml
            print(yaml.dump(config.to_dict(), default_flow_style=False))
        except ImportError:
            print("YAML format requires PyYAML package", file=sys.stderr)
            sys.exit(1)
    elif format_type == "env":
        for key, value in config.to_dict().items():
            if isinstance(value, list):
                value = ",".join(str(v) for v in value)
            print(f"DHRUVA_ENTERPRISE_{key.upper()}={value}")


def test_metrics():
    """Test metrics collection."""
    print("Testing metrics collection...")
    
    config = PluginConfig.from_env()
    metrics = MetricsCollector(config=config.to_dict())
    
    # Test request tracking
    print("Testing request tracking...")
    request_id = metrics.start_request("test_customer", "test_app", "/test", "test_service")
    metrics.end_request(request_id, 200)
    
    # Test component tracking
    print("Testing component tracking...")
    request_id = metrics.start_request("test_customer", "test_app", "/test", "test_service")
    metrics.start_component(request_id, "nmt")
    metrics.end_component(request_id, "nmt", success=True)
    metrics.end_request(request_id, 200)
    
    # Test service metrics
    print("Testing service metrics...")
    metrics.service_request("nmt", "test_customer", "test_app")
    metrics.nmt_chars("test_customer", "test_app", "en", "hi", 100)
    
    # Test system metrics
    print("Testing system metrics...")
    metrics.set_active_tenants(2)
    metrics.set_service_count("nmt", 1)
    
    print("✅ Metrics collection test completed successfully")
    print(f"Metrics available at: {config.metrics_path}")


def validate_config():
    """Validate configuration."""
    print("Validating configuration...")
    
    try:
        config = PluginConfig.from_env()
        
        # Check required settings
        if not config.customers:
            print("❌ No customers configured")
            return False
        
        if not config.apps:
            print("❌ No apps configured")
            return False
        
        # Check SLA targets
        if config.availability_target <= 0 or config.availability_target > 100:
            print("❌ Invalid availability target")
            return False
        
        if config.response_time_target <= 0:
            print("❌ Invalid response time target")
            return False
        
        if config.throughput_target <= 0:
            print("❌ Invalid throughput target")
            return False
        
        print("✅ Configuration is valid")
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False


def generate_config(output_file: str = None):
    """Generate sample configuration."""
    sample_config = {
        "enabled": True,
        "debug": False,
        "customers": ["customer1", "customer2"],
        "apps": ["app1", "app2"],
        "default_customer": "default",
        "default_app": "default",
        "metrics_path": "/enterprise/metrics",
        "health_path": "/enterprise/health",
        "collect_system_metrics": True,
        "collect_gpu_metrics": True,
        "collect_db_metrics": True,
        "availability_target": 100.0,
        "response_time_target": 1.0,
        "throughput_target": 20.0,
        "max_completed_requests": 1000,
        "metrics_update_interval": 10,
        "system_metrics_interval": 5
    }
    
    config_text = json.dumps(sample_config, indent=2)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(config_text)
        print(f"Sample configuration written to {output_file}")
    else:
        print("Sample Configuration:")
        print("=" * 20)
        print(config_text)


if __name__ == "__main__":
    main()
