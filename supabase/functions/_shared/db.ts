// Supabase data access. Uses the service-role key (auto-injected into Edge
// Functions), which bypasses RLS — so every query is explicitly scoped by user_id.

import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.7";
import { Labels, Person, UserState } from "./types.ts";

const supabase = createClient(
  Deno.env.get("SUPABASE_URL")!,
  Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!,
);

/** Cosine-similarity cutoff for "is this someone I already know?" (was distance < 0.3). */
export const SIM_THRESHOLD = 0.7;

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

/** Nearest remembered person for this user, or null if none clears the threshold. */
export async function matchPerson(
  userId: number,
  embedding: number[],
): Promise<Person | null> {
  const { data, error } = await supabase.rpc("match_person", {
    query_embedding: embedding,
    match_user_id: userId,
    match_threshold: SIM_THRESHOLD,
    match_count: 1,
  });
  if (error) throw error;
  return data?.[0] ?? null;
}

interface PersonInput {
  name: string | null;
  note: string;
  labels: Labels;
  embedding: number[];
}

export async function insertPerson(userId: number, p: PersonInput): Promise<void> {
  const { error } = await supabase.from("people").insert({
    user_id: userId,
    name: p.name,
    note: p.note,
    labels: p.labels,
    embedding: p.embedding,
  });
  if (error) throw error;
}

export async function updatePerson(id: string, p: PersonInput): Promise<void> {
  const { error } = await supabase
    .from("people")
    .update({ name: p.name, note: p.note, labels: p.labels, embedding: p.embedding })
    .eq("id", id);
  if (error) throw error;
}
