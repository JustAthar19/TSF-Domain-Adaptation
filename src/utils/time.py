from time import perf_counter

def convert_seconds(seconds):
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return f"{int(hours)}:{int(minutes)}:{int(seconds)}"


# def time_block(fn, label=""):
#     start = perf_counter()
#     result = fn()
#     elapsed = perf_counter() - start
#     print(f"{label} took: {elapsed} seconds to execute")
    