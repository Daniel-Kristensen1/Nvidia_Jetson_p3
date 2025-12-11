"""
Jetson Stats Testing Module
This module tests the jtop metrics and prints all available keys
"""

import time
from jtop import jtop


def test_jtop_metrics():
    """
    Test jtop metrics and print all available keys and their values.
    This helps identify what metrics are available for monitoring.
    """
    print("=" * 80)
    print("JETSON STATS - Testing jtop Metrics")
    print("=" * 80)
    
    try:
        with jtop() as jetson:
            print("\n✓ Successfully connected to jtop\n")
            
            # Get initial stats to display available keys
            stats = jetson.stats
            print(f"dict of stats: {stats} " )
            print("Available Keys in jetson.stats:")
            print("-" * 80)
            
            # Print all available keys with their types and raw values
            all_keys = sorted(stats.keys())
            gpu_related_keys = []
            
            for key in all_keys:
                value = stats[key]
                value_type = type(value).__name__
                
                # Handle datetime objects specially
                if value_type == 'datetime' or value_type == 'timedelta':
                    value_str = str(value)
                else:
                    value_str = str(value)
                
                    # Use repr to show the raw representation (including types like timedelta)
                    value_repr = repr(value)
                    print(f"  • {key:<20} = {value_repr} (type: {value_type})")
                
                # Track GPU-related keys
                if 'gpu' in key.lower() or 'gr3d' in key.lower() or 'gfxclk' in key.lower():
                    gpu_related_keys.append(key)
            
            print("\n" + "-" * 80)
            print(f"Total available metrics: {len(all_keys)}\n")
            
            # Print GPU-related keys specifically
            if gpu_related_keys:
                print("GPU-Related Keys Found:")
                for key in gpu_related_keys:
                        print(f"  → {key}: {repr(stats[key])}")
                print()
            else:
                print("⚠ No GPU-related keys found with common names")
                print("  Checking all keys that might relate to GPU usage...")
                print()
            
            # Now test with real-time updates
            print("Real-time Metrics Test (10 samples):")
            print("-" * 80)
            print(
                f"{'Time':<10} | "
                f"{'GPU%':<6} | "
                f"{'GPU_MHz':<8} | "
                f"{'CPU1%':<6} | "
                f"{'CPU2%':<6} | "
                f"{'CPU3%':<6} | "
                f"{'CPU4%':<6} | "
                f"{'RAM_GB':<8} | "
                f"{'Temp_GPU':<8} | "
                f"{'Temp_CPU':<8}"
            )
            print("-" * 80)
            
            for i in range(10):
                stats = jetson.stats
                
                # Try multiple possible GPU key names (varies by Jetson model)
                gpu_load = stats.get("GR3D", None)
                if gpu_load is None:
                    gpu_load = stats.get("gpu", None)
                if gpu_load is None:
                    gpu_load = stats.get("GR3D_FREQ", None)
                gpu_load = gpu_load if gpu_load is not None else "N/A"
                
                gpu_clock = stats.get("GPU", "N/A")
                if gpu_clock == "N/A":
                    gpu_clock = stats.get("gpu_freq", "N/A")
                
                cpu1 = stats.get("CPU1", "N/A")
                cpu2 = stats.get("CPU2", "N/A")
                cpu3 = stats.get("CPU3", "N/A")
                cpu4 = stats.get("CPU4", "N/A")
                ram_kb = stats.get("RAM", 0)
                ram_gb = ram_kb / (1024 * 1024) if ram_kb else 0
                temp_gpu = stats.get("Temp GPU", "N/A")
                temp_cpu = stats.get("Temp CPU", "N/A")
                
                # Format GPU load and clock
                gpu_load_str = f"{gpu_load}%" if isinstance(gpu_load, (int, float)) else gpu_load
                gpu_clock_str = f"{gpu_clock}" if gpu_clock != "N/A" else gpu_clock
                temp_gpu_str = f"{temp_gpu}°C" if isinstance(temp_gpu, (int, float)) else temp_gpu
                temp_cpu_str = f"{temp_cpu}°C" if isinstance(temp_cpu, (int, float)) else temp_cpu
                
                print(
                    f"{i+1:<10} | "
                    f"{str(gpu_load_str):<6} | "
                    f"{str(gpu_clock_str):<8} | "
                    f"{str(cpu1):<6} | "
                    f"{str(cpu2):<6} | "
                    f"{str(cpu3):<6} | "
                    f"{str(cpu4):<6} | "
                    f"{ram_gb:<8.2f} | "
                    f"{str(temp_gpu_str):<8} | "
                    f"{str(temp_cpu_str):<8}"
                )
                
                time.sleep(0.5)  # Wait 0.5 seconds between samples
            
            print("-" * 80)
            print("\n✓ Test completed successfully!")
            
    except Exception as e:
        print(f"\n✗ Error occurred: {type(e).__name__}: {e}")
        print("\nNote: This script must be run on a Jetson device with jtop installed.")
        print("Install jtop with: pip install jetson-stats")


def get_available_metrics():
    """
    Returns a dictionary of available metrics from jtop.
    Useful for programmatic access to what metrics are available.
    """
    try:
        with jtop() as jetson:
            return dict(jetson.stats)
    except Exception as e:
        print(f"Error getting metrics: {e}")
        return {}


if __name__ == "__main__":
    test_jtop_metrics()
