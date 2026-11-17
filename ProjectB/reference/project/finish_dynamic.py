"""Wait for the current development matrix, then finish without manual selection."""
import time
from full_io import settings
from run_dynamic import run

root,cfg=settings()
deadline=time.monotonic()+1800
while not (root/'results/dynamic_development_training.json').exists():
    if time.monotonic()>deadline:
        raise TimeoutError('Development did not complete within 30 minutes; inspect retained logs')
    time.sleep(5)
for phase in ('select','final','freeze','evaluate','analysis','verify','report'):
    run(root,phase)
