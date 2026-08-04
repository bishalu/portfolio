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
