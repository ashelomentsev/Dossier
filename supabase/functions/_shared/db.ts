// Supabase data access. Uses the service-role key (auto-injected into Edge
// Functions), which bypasses RLS — so every query is explicitly scoped by user_id.

import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.7";
import { Labels, Person, UserState } from "./types.ts";

const supabase = createClient(
  Deno.env.get("SUPABASE_URL")!,
  Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!,
);

/**
 * Cosine-similarity cutoff for matching purely on content, when there's no name
 * to lean on (or the nearest person has a different name).
 */
export const SIM_THRESHOLD = 0.7;

/**
 * Lower cutoff applied when the candidate shares the extracted name. A name is a
 * strong hint, so we accept a weaker content match — but still require *some*
 * overlap so a brand-new person who happens to share a name (another "Julia")
 * isn't merged into the wrong record. This is the main knob: raise it to split
 * more aggressively, lower it to merge more aggressively.
 */
export const NAME_SIM_THRESHOLD = 0.5;

/** How many candidates to pull back for the app-side selection. */
const MATCH_CANDIDATES = 5;

/** Ensure a user row exists and return its current state. */
export async function ensureUser(userId: number): Promise<UserState> {
  const { data, error } = await supabase
    .from("users")
    .upsert({ user_id: userId }, { onConflict: "user_id", ignoreDuplicates: false })
    .select("user_id, storage_mode, search_mode")
    .single();
  if (error) throw error;
  return data as UserState;
}

export async function setSearchMode(userId: number, on: boolean): Promise<void> {
  const { error } = await supabase
    .from("users")
    .update({ search_mode: on })
    .eq("user_id", userId);
  if (error) throw error;
}

/**
 * Decide which remembered person (if any) the new note refers to.
 *
 * Strategy:
 *  - Pull the candidate pool: everyone sharing the extracted name, plus anyone
 *    above SIM_THRESHOLD on content, nearest-first.
 *  - If the name matches one or more people, content picks *which* one (the most
 *    similar) and it merges only if that similarity clears NAME_SIM_THRESHOLD —
 *    so a different person with the same name becomes a new record instead.
 *  - With no name match, fall back to a pure content match above SIM_THRESHOLD.
 */
export async function matchPerson(
  userId: number,
  embedding: number[],
  name?: string | null,
): Promise<Person | null> {
  const { data, error } = await supabase.rpc("match_person", {
    query_embedding: embedding,
    match_user_id: userId,
    match_threshold: SIM_THRESHOLD,
    match_count: MATCH_CANDIDATES,
    match_name: name ?? null,
  });
  if (error) throw error;

  const candidates = (data ?? []) as Person[];
  const norm = (s?: string | null) => (s ?? "").trim().toLowerCase();
  // Candidates are nearest-first, so the first same-name entry is also the most
  // content-similar of the same-name people.
  const nameMatches = name ? candidates.filter((c) => norm(c.name) === norm(name)) : [];

  let chosen: Person | null = null;
  let reason = "no-match";
  if (nameMatches.length > 0) {
    const best = nameMatches[0];
    if ((best.similarity ?? 0) >= NAME_SIM_THRESHOLD) {
      chosen = best;
      reason = "name+content";
    } else {
      reason = "same-name-too-different"; // -> new distinct person
    }
  } else {
    const nearest = candidates[0];
    if (nearest && (nearest.similarity ?? 0) > SIM_THRESHOLD) {
      chosen = nearest;
      reason = name ? "content (no same-name)" : "content";
    }
  }

  console.log("matchPerson", {
    userId,
    queryName: name ?? null,
    candidates: candidates.length,
    sameName: nameMatches.length,
    bestSameName: nameMatches[0]
      ? { name: nameMatches[0].name, sim: nameMatches[0].similarity }
      : null,
    nearest: candidates[0] ? { name: candidates[0].name, sim: candidates[0].similarity } : null,
    nameThreshold: NAME_SIM_THRESHOLD,
    simThreshold: SIM_THRESHOLD,
    reason,
    matchedId: chosen?.id ?? null,
  });

  return chosen;
}

/**
 * Permissive floor for recall. A short query ("a girl who sold Swiss watches")
 * sits well below the 0.7 capture threshold against a full note, so search uses
 * a low floor and returns the single nearest person above it.
 */
export const RECALL_THRESHOLD = 0.2;

/** Find the closest person to a recall query (nearest-first, low floor). */
export async function searchPerson(
  userId: number,
  embedding: number[],
  name?: string | null,
): Promise<Person | null> {
  const { data, error } = await supabase.rpc("match_person", {
    query_embedding: embedding,
    match_user_id: userId,
    match_threshold: RECALL_THRESHOLD,
    match_count: 1,
    match_name: name ?? null,
  });
  if (error) throw error;
  const candidates = (data ?? []) as Person[];
  return candidates[0] ?? null; // RPC orders nearest-first
}

interface PersonInput {
  name: string | null;
  note: string;
  labels: Labels;
  embedding: number[];
}

/** Insert a new person and return its generated id (used to address later edits). */
export async function insertPerson(userId: number, p: PersonInput): Promise<string> {
  const { data, error } = await supabase
    .from("people")
    .insert({
      user_id: userId,
      name: p.name,
      note: p.note,
      labels: p.labels,
      embedding: p.embedding,
    })
    .select("id")
    .single();
  if (error) throw error;
  return (data as { id: string }).id;
}

/** Fetch a single person, scoped to its owner so one user can't read another's record. */
export async function getPerson(id: string, userId: number): Promise<Person | null> {
  const { data, error } = await supabase
    .from("people")
    .select("id, name, note, labels")
    .eq("id", id)
    .eq("user_id", userId)
    .maybeSingle();
  if (error) throw error;
  return (data as Person) ?? null;
}

// Scoped by user_id as well as id: a belt-and-suspenders against editing a record
// the chat doesn't own, even though id arrives via a (non-forgeable) reply.
export async function updatePerson(
  id: string,
  userId: number,
  p: PersonInput,
): Promise<void> {
  const { error } = await supabase
    .from("people")
    .update({ name: p.name, note: p.note, labels: p.labels, embedding: p.embedding })
    .eq("id", id)
    .eq("user_id", userId);
  if (error) throw error;
}
