from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from alerting import detect_changes_and_alert
from datetime import datetime
import time

def start_scheduler():
    """Start background scheduler for hourly checks"""
    
    scheduler = BackgroundScheduler()
    
    # Run every hour
    scheduler.add_job(
        detect_changes_and_alert,
        trigger=IntervalTrigger(hours=1),
        id='sentiment_check',
        name='Hourly sentiment analysis',
        replace_existing=True
    )
    
    scheduler.start()
    
    print("✓ Scheduler started - checking sentiment every hour")
    print(f"Next run: {scheduler.get_job('sentiment_check').next_run_time}")
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        scheduler.shutdown()
        print("✓ Scheduler stopped")

if __name__ == "__main__":
    start_scheduler()