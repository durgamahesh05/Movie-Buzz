#!/usr/bin/env python3
"""
🔑 Minimal MongoDB Connection Test
Test connectivity using the MONGODB_URI from your environment.
"""

import os

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

CONNECTION_STRING = os.getenv("MONGODB_URI", "").strip()

print("=" * 70)
print("🧪 Testing MongoDB Connection")
print("=" * 70)

if not CONNECTION_STRING:
    print("\n❌ MONGODB_URI is not set.")
    print("   Add your MongoDB Atlas URI to the environment or .env before running this script.")
    raise SystemExit(1)

print(f"\n📍 Connection String: {CONNECTION_STRING[:50]}...")

try:
    print("\n🔗 Connecting to MongoDB Atlas...")
    
    client = MongoClient(
        CONNECTION_STRING,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000
    )
    
    # Test connection
    print("✅ Connected!")
    
    # List databases
    databases = client.list_database_names()
    print(f"\n📊 Databases available:")
    for db in databases:
        print(f"   • {db}")
    
    # Try to access moviebuzz database
    db = client['moviebuzz']
    collections = db.list_collection_names()
    print(f"\n📁 Collections in 'moviebuzz':")
    if collections:
        for col in collections:
            print(f"   • {col}")
    else:
        print("   (empty - will be populated on first load)")
    
    # Ping admin database
    admin_db = client['admin']
    admin_db.command('ping')
    print("\n✅ Authentication successful!")
    
    print("\n" + "=" * 70)
    print("✨ SUCCESS! MongoDB is properly configured")
    print("=" * 70)
    print("\nYou can now use:")
    print("  from mongo_connection import MongoDBConnection")
    print("  conn = MongoDBConnection()")
    print("  conn.connect_env()  # Uses MONGODB_URI from .env")
    
    client.close()
    
except ServerSelectionTimeoutError:
    print("\n❌ Connection Timeout")
    print("   Check 1: Is cluster online in Atlas?")
    print("   Check 2: Is network access whitelisted (0.0.0.0/0)?")
    print("   Check 3: Is internet connection working?")
    
except ConnectionFailure as e:
    print(f"\n❌ Connection Failed: {e}")
    print("   Check: Username and password are correct?")
    
except Exception as e:
    error_msg = str(e)
    
    if "authentication failed" in error_msg.lower():
        print(f"\n❌ Authentication Failed")
        print("   ❌ Bad username/password")
        print("   ✅ DELETE and CREATE NEW user:")
        print("      Username: admin")
        print("      Password: Moviebuzz@spm")
        print("      Role: Read and write to any database")
        
    elif "bad auth" in error_msg.lower():
        print(f"\n❌ Bad Authentication")
        print("   Make sure:")
        print("   1. Username is exactly: admin")
        print("   2. Password is exactly: Moviebuzz@spm")
        print("   3. No typos in connection string")
        
    else:
        print(f"\n❌ Error: {e}")
        print("   Try restarting terminal or VS Code")

print()
