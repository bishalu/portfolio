/**
 * Make the real retrieval code runnable from a plain Node script.
 *
 * WHY THIS EXISTS: the eval harness has to measure the SHIPPING retriever, not a
 * copy of it. A reimplementation of BM25 in the harness would score itself and
 * agree with itself forever, which is the exact failure the harness is being
 * built to prevent. So `src/lib/naam/match.ts` is transpiled and imported as-is.
 *
 * Node 20 has no `--experimental-strip-types`, and match.ts uses the `@/*` path
 * alias, so neither a bare import nor a loader flag works. TypeScript is already
 * a dependency; `transpileModule` strips the types in a few milliseconds and the
 * one alias in play is rewritten by hand. No new packages, no bundler.
 *
 * Output goes to node_modules/.cache — already ignored, already disposable, and
 * never mistaken for source.
 */
import { createRequire } from 'node:module'
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
export const REPO = resolve(HERE, '../../..')

const require = createRequire(resolve(REPO, 'package.json'))

/** Source file → the name it gets in the cache dir. */
const MODULES = [
  ['src/types/naam.ts', 'naam-types.mjs'],
  ['src/lib/naam/match.ts', 'naam-match.mjs'],
]

/** The single alias match.ts actually uses. Kept explicit — a regex over every
 *  possible alias would silently rewrite things nobody checked. */
const ALIAS = [[/from '@\/types\/naam'/g, "from './naam-types.mjs'"]]

/**
 * Transpile the pure retrieval modules and return the imported namespace.
 * Pure means: no I/O, no globals, no framework. match.ts is documented as
 * isomorphic and dependency-free, which is what makes this safe.
 */
export async function loadMatch() {
  const ts = require('typescript')
  const outDir = resolve(REPO, 'node_modules/.cache/naam-eval')
  mkdirSync(outDir, { recursive: true })

  for (const [src, out] of MODULES) {
    const source = readFileSync(resolve(REPO, src), 'utf8')
    const { outputText } = ts.transpileModule(source, {
      compilerOptions: {
        target: ts.ScriptTarget.ES2022,
        module: ts.ModuleKind.ESNext,
        // Types are erased, not checked. `tsc` and the editor own correctness;
        // this only has to produce runnable JS.
        isolatedModules: true,
      },
      fileName: src,
    })
    let js = outputText
    for (const [re, to] of ALIAS) js = js.replace(re, to)
    writeFileSync(resolve(outDir, out), js)
  }

  return import(pathToFileURL(resolve(outDir, 'naam-match.mjs')).href)
}

/**
 * The POOL BUILDER, which is what production actually uses and what
 * `loadMatch()` does not reach.
 *
 * The distinction matters and it was found the expensive way: run.mjs scores
 * `match.retrieve()`, but a live ask goes readAsk -> pool() -> rank(). They are
 * different functions with different failure modes — retrieve() alone reaches
 * 21.9% of the corpus and returns nothing for bare form words like "short",
 * while the production path handles those and never returned empty in 82
 * queries. Success@3 is a real number about a function the page does not use to
 * assemble the pool.
 *
 * ask.ts is not pure — it imports copy, tray and prompt for strings, caps and a
 * thesaurus accessor. Those are stubbed rather than transpiled: none of them
 * influences WHICH ids come back, and pulling the real ones would drag React
 * and the DOM into a Node script. The thesaurus is passed through for real,
 * because it does change the answer.
 */
export async function loadAsk(thesaurus = {}) {
  const ts = require('typescript')
  const outDir = resolve(REPO, 'node_modules/.cache/naam-eval')
  mkdirSync(outDir, { recursive: true })

  for (const [src, out] of [...MODULES, ['src/lib/naam/ask.ts', 'naam-ask.mjs']]) {
    const source = readFileSync(resolve(REPO, src), 'utf8')
    const { outputText } = ts.transpileModule(source, {
      compilerOptions: {
        target: ts.ScriptTarget.ES2022,
        module: ts.ModuleKind.ESNext,
        isolatedModules: true,
      },
      fileName: src,
    })
    let js = outputText
      .replace(/from '@\/types\/naam'/g, "from './naam-types.mjs'")
      .replace(/from '@\/lib\/naam\/match'/g, "from './naam-match.mjs'")
      .replace(/from '\.\/match'/g, "from './naam-match.mjs'")
      .replace(/from '@\/generated\/naam-facts'/g, "from './naam-facts.mjs'")
      .replace(/from '\.\/copy'/g, "from './naam-copy.mjs'")
      .replace(/from '@\/lib\/naam\/copy'/g, "from './naam-copy.mjs'")
      .replace(/from '\.\/tray'/g, "from './naam-tray.mjs'")
      .replace(/from '@\/lib\/naam\/tray'/g, "from './naam-tray.mjs'")
      .replace(/from '\.\/prompt'/g, "from './naam-prompt.mjs'")
      .replace(/from '@\/lib\/naam\/prompt'/g, "from './naam-prompt.mjs'")
    writeFileSync(resolve(outDir, out), js)
  }

  globalThis.__NAAM_THESAURUS = thesaurus
  writeFileSync(
    resolve(outDir, 'naam-facts.mjs'),
    'export const NAAM_COUNTS={total:6715};export const NAAM_FEATURED=[];export const NAAM_PROVENANCE={};\n',
  )
  writeFileSync(
    resolve(outDir, 'naam-copy.mjs'),
    'export const NAAM_COPY=new Proxy({},{get:()=>new Proxy({},{get:()=>""})});\n',
  )
  writeFileSync(
    resolve(outDir, 'naam-tray.mjs'),
    'export const PICK_MAX=3;export const NAAM_TRAY={};export function readTray(){return[]}\n' +
      'export function currentThesaurus(){return globalThis.__NAAM_THESAURUS||{}}\n',
  )
  writeFileSync(
    resolve(outDir, 'naam-prompt.mjs'),
    'export const NAAM_DEAL_SMALL=6;export const NAAM_DEAL_LARGE=8;export const NAAM_MAX_PICKS=8;\n' +
      'export function snapDeal(n){return n>=8?8:n>=6?6:n}\n',
  )

  return {
    ask: await import(pathToFileURL(resolve(outDir, 'naam-ask.mjs')).href),
    match: await import(pathToFileURL(resolve(outDir, 'naam-match.mjs')).href),
  }
}

/** The built query thesaurus. `{}` when it has not been generated yet, which is
 *  a valid state — retrieval works without it, just with the gaps. */
export function loadThesaurus() {
  const path = resolve(REPO, 'public/naam/thesaurus.json')
  try {
    return JSON.parse(readFileSync(path, 'utf8'))
  } catch {
    return {}
  }
}

/** The 2,098 core rows, straight off disk — the same artifact the browser and
 *  the Lambda both fetch. Built by scripts/naam/build-dataset.mjs. */
export function loadRows() {
  const path = resolve(REPO, 'public/naam/names-core.json')
  const rows = JSON.parse(readFileSync(path, 'utf8'))
  if (!Array.isArray(rows) || rows.length === 0) {
    throw new Error(`no rows in ${path} — run \`npm run naam:build\` first`)
  }
  return rows
}
