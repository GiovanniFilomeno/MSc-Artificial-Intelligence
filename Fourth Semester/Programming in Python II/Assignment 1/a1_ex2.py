import time
from collections import deque

class RateLimiter:
    def __init__(self, max_calls: int, window_seconds: float):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.timestamps = deque()
        
    def __call__(self) -> bool:
        # Current time
        now = time.time()
        # Older timestamps --> initially >=, changed based on the unittest 
        while self.timestamps and (now - self.timestamps[0]) > self.window_seconds:
            self.timestamps.popleft()
        
        # Check the max calls
        if len(self.timestamps) < self.max_calls:
            self.timestamps.append(now)
            return True
        else:
            return False


if __name__ == "__main__":
    # Copy-Paste from exercise
    limiter = RateLimiter(3, 5.0)
    
    for i in range(5):
        time.sleep(1)
        success = limiter()
        print(f"Call #{i+1} => {success}")
        
    for i in range(3):
        time.sleep(2)
        success = limiter()
        print(f"Call #{i+6} => {success}")
