# Supabase setup for Synera 2.0

## Step 1 — Create Supabase project

1. Go to https://supabase.com → **New Project**
2. **Name:** `synera-arogyalink`
3. **Database password:** Save this — it goes in `.env`
4. **Region:** `Southeast Asia (Singapore)` — closest to India
5. Wait for project to provision (~2 minutes)

## Step 2 — Enable pgvector

In Supabase **SQL Editor**, run:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

(Or run the full `supabase/schema.sql` in one go — it includes this.)

## Step 3 — Run full schema

1. Open **SQL Editor** → **New query**
2. Copy the entire contents of **`supabase/schema.sql`**
3. Paste and click **Run**

This creates: `patients`, `vitals_history`, `alert_events`, `medical_knowledge`, `clinical_cases`, and the vector search functions.

## Step 4 — Get credentials

**From Project Settings → API:**

- `SUPABASE_URL` = **Project URL** (e.g. `https://xxxx.supabase.co`)
- `SUPABASE_ANON_KEY` = anon/public key
- `SUPABASE_SERVICE_KEY` = **service_role** key (use this for backend)

**From Project Settings → Database:**

- `DATABASE_URL` = **Connection string** (URI format; **Transaction pooler** recommended)
  - Example: `postgresql://postgres.[ref]:[password]@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres`

Add these to your `.env` (copy from `.env.example` first).

## Step 5 — Validate (optional)

```bash
python scripts/setup/init_supabase.py
```

This checks that `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` are set and that the `patients` table exists.
