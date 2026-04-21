import time

def convert_seconds(seconds):
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return f"{int(hours)}:{int(minutes)}:{int(seconds)}"