-- Name-aware matching. Whole-note embeddings alone are too blunt to recognize
-- the same person described differently (e.g. two notes about "Julia" can sit at
-- ~0.48 cosine, under the 0.7 threshold), which produced duplicate records. But a
-- name alone is not enough either: the user may know several people with the same
-- name, so content still has to decide *which* one (or whether it's a new person).
--
-- This function just returns the candidate pool: every person whose name matches
-- (regardless of distance) plus anyone above the similarity floor, ordered
-- nearest-first. The Edge Function (matchPerson in db.ts) does the final
-- selection: among same-name candidates it picks the most content-similar and
-- only merges if that similarity clears a name-match floor; otherwise it treats
-- the note as a new, distinct person.
--
-- match_name is optional (defaults to null) so the signature stays flexible.
--
-- Drop the original 4-arg signature first: adding a parameter creates a *new*
-- overload rather than replacing it, which makes 4-arg calls ambiguous.
drop function if exists match_person(vector(1536), bigint, float, int);

create or replace function match_person(
  query_embedding vector(1536),
  match_user_id   bigint,
  match_threshold float,
  match_count     int,
  match_name      text default null
)
returns table (id uuid, name text, note text, labels jsonb, similarity float)
language sql stable as $$
  select p.id, p.name, p.note, p.labels,
         1 - (p.embedding <=> query_embedding) as similarity
  from people p
  where p.user_id = match_user_id
    and (
      (match_name is not null and lower(p.name) = lower(match_name))
      or 1 - (p.embedding <=> query_embedding) > match_threshold
    )
  order by p.embedding <=> query_embedding  -- nearest first; app makes the call
  limit match_count;
$$;
