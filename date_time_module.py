from datetime import datetime
import pytz

def get_current_datetime():
    """
    Returns current date and time with timezone info.
    """
    # Change to your timezone, example Asia/Kolkata for India
    timezone = pytz.timezone("Asia/Kolkata")

    now = datetime.now(timezone)

    return {
        "date": now.strftime("%d-%m-%Y"),
        "time": now.strftime("%H:%M:%S"),
        "day": now.strftime("%A"),
        "timezone": str(timezone),
        "full_timestamp": now.strftime("%d-%m-%Y %H:%M:%S %Z %z")
    }


# Demo
if __name__ == "__main__":
    data = get_current_datetime()
    print("\n--- Current Date & Time Info ---")
    print(f"Date        : {data['date']}")
    print(f"Time        : {data['time']}")
    print(f"Day         : {data['day']}")
    print(f"Timezone    : {data['timezone']}")
    print(f"Full Format : {data['full_timestamp']}")

