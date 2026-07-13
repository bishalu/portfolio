/**
 * Replay fixtures for the Vibeset demo proxy — real responses captured from
 * the production API (2026-07-12), served whenever the live API is cold,
 * slow, or unreachable. The widget labels these results REPLAY.
 */

export type SlimSuggestion = {
  display: string
  type: string
  artist?: string
  track?: string
  genre?: string
  tempo?: number
}

export type SlimTrack = {
  track: string
  artist: string
  genre: string
  mood: string
  energy: string
  tempo: number | null
  year: number | null
}

export const FIXTURE_SUGGESTIONS: Record<string, SlimSuggestion[]> = {
  di: [
    { display: 'Disclosure', type: 'artist', artist: 'Disclosure', track: 'Latch', genre: 'Electronic', tempo: 124 },
    { display: 'Diplo', type: 'artist', artist: 'Diplo', genre: 'Electronic' },
  ],
  fr: [
    { display: 'Fred again..', type: 'artist', artist: 'Fred again..', track: 'Turn On The Lights again..', genre: 'Electronic', tempo: 128 },
  ],
  bi: [
    { display: 'Bicep', type: 'artist', artist: 'Bicep', track: 'Higher Level (Bicep Remix)', genre: 'Electronic', tempo: 128 },
  ],
  de: [
    { display: 'deep house', type: 'genre', genre: 'deep house' },
    { display: 'Deadmau5', type: 'artist', artist: 'Deadmau5', genre: 'Electronic' },
  ],
  me: [
    { display: 'melodic techno', type: 'genre', genre: 'melodic techno' },
    { display: 'Melody', type: 'track', artist: 'aespa', track: 'Melody', genre: 'Pop', tempo: 120 },
  ],
}

export const FIXTURE_TRACKS: SlimTrack[] = [
  { track: 'Onderhuids', artist: 'Eelke Kleijn', genre: 'Deep House', mood: 'Atmospheric', energy: 'high', tempo: 122, year: 2019 },
  { track: 'Neverland (From Japan)', artist: 'Anyma', genre: 'Melodic Techno', mood: 'Ethereal', energy: 'high', tempo: 125, year: 2025 },
  { track: 'Allein Allein', artist: 'Alok', genre: 'Melodic Techno', mood: 'Euphoric', energy: 'high', tempo: 125, year: 2024 },
  { track: 'Inside Your Mind', artist: 'Innellea', genre: 'Melodic Techno', mood: 'Introspective', energy: 'medium', tempo: 124, year: 2025 },
  { track: 'Act Of God', artist: 'Layton Giordani', genre: 'Melodic Techno', mood: 'Energetic', energy: 'high', tempo: 125, year: 2025 },
  { track: 'Latch', artist: 'Disclosure', genre: 'Electronic', mood: 'Romantic', energy: 'high', tempo: 124, year: 2012 },
]

export function fixtureSuggestions(query: string): SlimSuggestion[] {
  const key = query.trim().toLowerCase().slice(0, 2)
  return FIXTURE_SUGGESTIONS[key] ?? FIXTURE_SUGGESTIONS['me']
}

export function fixtureTracks(limit: number): SlimTrack[] {
  return FIXTURE_TRACKS.slice(0, Math.max(1, Math.min(limit, FIXTURE_TRACKS.length)))
}
