from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from full_pipeline import run_full_pipeline
from datetime import datetime
import time

def start_scheduler():
    """Start background scheduler for hourly pipeline checks"""
    
    scheduler = BackgroundScheduler()
    
    scheduler.add_job(
        run_full_pipeline,
        trigger=IntervalTrigger(hours=1),
        id='sentiment_pipeline',
        name='Full sentiment pipeline',
        replace_existing=True
    )
    
    scheduler.start()
    
    print("✓ Scheduler started - running full pipeline every hour")
    print(f"Next run: {scheduler.get_job('sentiment_pipeline').next_run_time}")
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        scheduler.shutdown()
        print("✓ Scheduler stopped")

if __name__ == "__main__":
    start_scheduler()