import time
from datetime import datetime 
start = datetime.now()
a = sum(a**2 for a in range(1,1000000))

time = (datetime.now() - start).total_seconds()
print(time)
