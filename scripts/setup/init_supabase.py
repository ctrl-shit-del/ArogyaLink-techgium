"""
One-time setup script.
Run supabase/schema.sql manually in Supabase SQL Editor (Supabase does not expose raw SQL execution via API).
This script prints instructions and optionally validates connection.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def main():
    schema_path = os.path.join(os.path.dirname(__file__), "..", "..", "supabase", "schema.sql")
    if not os.path.exists(schema_path):
        print("[ERROR] supabase/schema.sql not found")
        sys.exit(1)

    print("=" * 60)
    print("SYNERA 2.0 — Supabase schema setup")
    print("=" * 60)
    print()
    print("Supabase does not expose a raw SQL execution RPC by default.")
    print("You must run the schema manually in the Supabase Dashboard:")
    print()
    print("  1. Go to your project: https://supabase.com/dashboard")
    print("  2. Open SQL Editor → New query")
    print("  3. Copy the contents of supabase/schema.sql")
    print("  4. Paste and click Run")
    print()
    print(f"  Schema file: {os.path.abspath(schema_path)}")
    print()

    try:
        from dotenv import load_dotenv
        load_dotenv()
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
        if url and key:
            from supabase import create_client
            client = create_client(url, key)
            # Quick validation: list tables or run a simple query
            r = client.table("patients").select("patient_id").limit(1).execute()
            print("[OK] Supabase connection valid. Table 'patients' exists.")
        else:
            print("[SKIP] SUPABASE_URL or SUPABASE_SERVICE_KEY not set. Set them in .env and run again to validate.")
    except Exception as e:
        print(f"[INFO] Could not validate connection: {e}")
        print("  Make sure .env has SUPABASE_URL and SUPABASE_SERVICE_KEY, then run schema in SQL Editor.")

    print()
    print("After running the schema, seed data: python scripts/setup/seed_data.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
