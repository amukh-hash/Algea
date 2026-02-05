
import uuid
import os
from datetime import datetime

class RunContext:
    def __init__(self):
        self.run_id = f"run_{datetime.utcnow().strftime('%Y%m%d')}_{str(uuid.uuid4())[:8]}"
        self.start_time = datetime.utcnow()
        
    def get_run_id(self):
        return self.run_id
