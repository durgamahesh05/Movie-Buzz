#!/usr/bin/env python3
"""
🔧 MovieBuzz MongoDB Atlas Interactive Setup
Guides you through MongoDB Atlas configuration step-by-step
"""

import os
import sys
from pathlib import Path

def print_header(text):
    print("\n" + "="*70)
    print(text)
    print("="*70 + "\n")

def print_step(num, text):
    print(f"📌 STEP {num}: {text}")
    print("-" * 70)

def main():
    print_header("🍃 MongoDB Atlas Setup Wizard")
    
    print("This wizard will help you set up MongoDB Atlas for MovieBuzz.\n")
    
    # Step 1: Check .env file
    print_step(1, "Check .env File")
    env_path = Path("../.env")
    
    if env_path.exists():
        print("✅ .env file found!")
        with open(env_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if "MONGODB_URI" in content:
                print("✅ MONGODB_URI already in .env")
            else:
                print("⚠️ MONGODB_URI not found in .env - we'll add it\n")
    else:
        print("❌ .env file not found - creating one...\n")
    
    # Step 2: Get MongoDB Atlas URI
    print_step(2, "Get Your MongoDB Atlas Connection String")
    print("""
Follow these steps:
1. Go to: https://www.mongodb.com/cloud/atlas
2. Create free account (if you don't have one)
3. Create M0 cluster (free tier)
4. Click "Connect" → "Drivers" → Select Python
5. Copy the connection string
    """)
    
    uri = input("Paste your MongoDB Atlas connection string: ").strip()
    
    if not uri.startswith("mongodb+srv://"):
        print("❌ Invalid URI format. Should start with: mongodb+srv://")
        return
    
    # Step 3: Verify URI format
    print_step(3, "Verify Connection String")
    if "@" in uri and ".mongodb.net" in uri:
        print("✅ URI format looks valid\n")
    else:
        print("⚠️ URI might be incomplete\n")
    
    # Step 4: Test connection
    print_step(4, "Test Connection")
    print("Testing connection to MongoDB Atlas...\n")
    
    try:
        from pymongo import MongoClient
        
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        
        print("✅ Connection successful!\n")
        
        # Get database info
        db = client['moviebuzz']
        collections = db.list_collection_names()
        
        print(f"📊 Database: moviebuzz")
        print(f"📁 Collections: {len(collections)} found")
        
        client.close()
        
    except Exception as e:
        print(f"❌ Connection failed: {e}\n")
        print("⚠️ Please check:")
        print("  1. Connection string is correct")
        print("  2. IP is whitelisted in Network Access (0.0.0.0/0)")
        print("  3. Database user was created")
        print("  4. Username/password are correct")
        return
    
    # Step 5: Update .env
    print_step(5, "Update .env File")
    
    try:
        env_file = Path("../.env")
        
        # Read existing content
        if env_file.exists():
            with open(env_file, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            content = ""
        
        # Update or add MONGODB_URI
        if "MONGODB_URI=" in content:
            # Replace existing
            lines = content.split('\n')
            new_lines = []
            for line in lines:
                if line.startswith("MONGODB_URI="):
                    new_lines.append(f"MONGODB_URI={uri}")
                else:
                    new_lines.append(line)
            content = '\n'.join(new_lines)
        else:
            # Add new
            content += f"\nMONGODB_URI={uri}\n"
        
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ .env file updated!\n")
        
    except Exception as e:
        print(f"❌ Error updating .env: {e}\n")
        print("📌 Manually add to .env:")
        print(f"MONGODB_URI={uri}\n")
    
    # Step 6: Verify setup
    print_step(6, "Verify Complete Setup")
    
    try:
        # Import after .env is set
        sys.path.insert(0, str(Path(__file__).parent))
        from mongo_connection import MongoDBConnection
        
        conn = MongoDBConnection()
        if conn.connect_env():
            stats = conn.get_stats()
            print("✅ Setup Complete!\n")
            print(f"📊 Status: {stats.get('status', 'unknown')}")
            print(f"📁 Database: {stats.get('database', 'unknown')}")
            print(f"📦 Collections: {stats.get('collections', [])}\n")
            
            conn.close()
            
            print("✨ You're ready to use MongoDB!")
            print("\nNext steps:")
            print("1. Add your CSV data: python migrate_to_mongodb.py")
            print("2. Use in your app:")
            print("   from mongo_connection import MongoDBConnection")
            print("   conn = MongoDBConnection()")
            print("   conn.connect_env()")
            print("   db = conn.get_db()\n")
        else:
            print("❌ Connection still failing. Check troubleshooting guide.\n")
            
    except Exception as e:
        print(f"⚠️ Verification error: {e}\n")
        print("This is okay - your connection string is saved in .env\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        sys.exit(1)
