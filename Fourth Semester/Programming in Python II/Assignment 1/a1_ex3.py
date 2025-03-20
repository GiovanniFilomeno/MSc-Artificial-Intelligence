import os
import time
from functools import wraps

def file_cache(cache_dir: str = "cache"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)

            filename_parts = [func.__name__] + [str(a) for a in args]
            file_name = "_".join(filename_parts) + "_.txt"
            file_path = os.path.join(cache_dir, file_name)

            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    cached_result = f.read()
                print(f"[CACHE] Using cached result for {func.__name__} with args={args}")
                return cached_result

            result = func(*args, **kwargs)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(str(result))
            return result
        return wrapper
    return decorator


# Copy-Paste from exercise
@file_cache(cache_dir="cache_txt")
def expensive_calculation(x: int, y: int) -> str:
    time.sleep(2)  # Simulates a long computation
    return f"The result is {x * y}"


if __name__ == '__main__':
    print(expensive_calculation(10, 20))
    # Second call with the same argument: should return immediately from cache.
    print(expensive_calculation(10, 20))
    # New argument => computation happens again.
    print(expensive_calculation(5, 15))
