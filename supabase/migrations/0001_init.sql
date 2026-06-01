-- Dossier — Phase 1 schema (multi-tenant, Supabase-canonical).
-- pgvector index over per-user "people" notes, plus per-user bot state.
-- Phase 2 (Obsidian/vault) columns are present but unused until a repo is connected.

create extension if not exists vector;

-- One row per Telegram user. Holds bot state and (later) the vault connection.
create table if not exists users (
  user_id            bigint primary key,                       -- Telegram chat id
  storage_mode       text not null default 'supabase'
                       check (storage_mode in ('supabase', 'vault')),
  vault_repo         text,                                     -- "owner/repo"      (Phase 2)
  gh_installation_id bigint,                                   -- GitHub App install (Phase 2)
  search_mode        boolean not null default false,           -- next voice note is a query
  created_at         timestamptz not null default now(),
  updated_at         timestamptz not null default now()
);

-- One row per remembered person, scoped to a user.
create table if not exists people (
  id          uuid primary key default gen_random_uuid(),
  user_id     bigint not null references users(user_id) on delete cascade,
  name        text,                                            -- extracted display name, if known
  note        text not null,                                   -- accumulated raw notes (merged over time)
  labels      jsonb not null default '{}'::jsonb,              -- extracted structured fields
  embedding   vector(1536),                                    -- text-embedding-3-small
  file_path   text,                                            -- vault path        (Phase 2)
  git_sha     text,                                            -- last synced blob  (Phase 2)
  source      text not null default 'bot'
                       check (source in ('bot', 'vault')),
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);

create index if not exists people_user_id_idx on people (user_id);
create index if not exists people_embedding_idx
  on people using hnsw (embedding vector_cosine_ops);

-- All access goes through the Edge Function using the service-role key, which
-- bypasses RLS. Enabling RLS with no policies blocks anon/authenticated keys,
-- keeping each user's data isolated by default.
alter table users  enable row level security;
alter table people enable row level security;

-- Nearest remembered person for a user, filtered by cosine similarity threshold.
create or replace function match_person(
  query_embedding vector(1536),
  match_user_id   bigint,
  match_threshold float,
  match_count     int
)
returns table (id uuid, name text, note text, labels jsonb, similarity float)
language sql stable as $$
  select p.id, p.name, p.note, p.labels,
         1 - (p.embedding <=> query_embedding) as similarity
  from people p
  where p.user_id = match_user_id
    and 1 - (p.embedding <=> query_embedding) > match_threshold
  order by p.embedding <=> query_embedding
  limit match_count;
$$;

-- Keep updated_at current on writes.
create or replace function touch_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create trigger people_touch before update on people
  for each row execute function touch_updated_at();

create trigger users_touch before update on users
  for each row execute function touch_updated_at();
