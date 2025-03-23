import os
import sqlite3
import sys
import gc
import time

print("Trying to free up database connections...")

# Force garbage collection to release any open database connections
gc.collect()

# Create dummy connection and close it
try:
    conn = sqlite3.connect('jooble_jobs.db')
    conn.close()
    print("Successfully connected and closed connection")
except Exception as e:
    print(f"Could not connect to database: {e}")

# Wait a moment
time.sleep(2)

# Try to rename the database file as a workaround
try:
    if os.path.exists('jooble_jobs.db'):
        # Try to rename the file
        if os.path.exists('jooble_jobs.db.bak'):
            os.remove('jooble_jobs.db.bak')
        os.rename('jooble_jobs.db', 'jooble_jobs.db.bak')
        print("Successfully renamed database file to jooble_jobs.db.bak")
        print("\nNow run: python test.py")
    else:
        print("Database file does not exist")
except Exception as e:
    print(f"Could not rename the database file: {e}")
    print("\nPlease close all Python processes and database-related applications, then try again.") 