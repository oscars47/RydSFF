import sys
from datetime import datetime, timezone

def unix_to_utc(ts: int) -> str:
    """
    Convert a Unix timestamp (seconds since epoch) to a UTC datetime string.
    """
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

## add another function to convert back
def utc_to_unix(utc_str: str) -> int:
    """
    Convert a UTC datetime string to a Unix timestamp (seconds since epoch).
    """
    dt = datetime.strptime(utc_str, "%Y-%m-%d %H:%M:%S UTC")
    dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python unix_to_utc.py <unix_timestamp_or_datetime_string>")
        sys.exit(1)

    input_arg = sys.argv[1]
    
    # Try to parse as Unix timestamp (integer)
    try:
        timestamp = int(input_arg)
        print(unix_to_utc(timestamp))
    except ValueError:
        # If not an integer, try parsing as datetime string
        try:
            # First try ISO 8601 format (e.g., 2026-01-02T09:54:47.865-05:00)
            dt = datetime.fromisoformat(input_arg)
            print(int(dt.timestamp()))
        except ValueError:
            # Fall back to the simple UTC format
            try:
                print(utc_to_unix(input_arg))
            except ValueError as e:
                print(f"Error: Could not parse input as Unix timestamp or datetime string")
                print("Supported formats:")
                print("  - Unix timestamp: 1704844800")
                print("  - ISO 8601: 2026-01-02T09:54:47.865-05:00")
                print("  - Simple UTC: 2024-01-10 00:00:00 UTC")
                sys.exit(1)
