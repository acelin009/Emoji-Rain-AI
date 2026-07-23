"""
FPS Counter for performance monitoring
"""

import time
from collections import deque


class FPSCounter:
    def __init__(self, window_size=30):
        self.timestamps = deque(maxlen=window_size)
        self._fps = 0
        self._last_update = time.time()
        self._update_interval = 0.5  # Update FPS display every 0.5s
    
    def tick(self):
        """Call this every frame"""
        self.timestamps.append(time.time())
        
        # Update FPS calculation periodically
        if time.time() - self._last_update >= self._update_interval:
            if len(self.timestamps) > 1:
                time_diff = self.timestamps[-1] - self.timestamps[0]
                self._fps = len(self.timestamps) / time_diff if time_diff > 0 else 0
            self._last_update = time.time()
        
        return self._fps
    
    @property
    def fps(self):
        return self._fps
    
    def reset(self):
        """Reset the counter"""
        self.timestamps.clear()
        self._fps = 0
        self._last_update = time.time()