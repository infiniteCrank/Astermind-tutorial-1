# ESM __dirname Bug in @astermind/astermind-synth

## Problem

The `@astermind/astermind-synth` library (v0.1.5) uses `__dirname` in an ESM (ECMAScript Module) file, which causes a `ReferenceError: __dirname is not defined` error.

### Error Details

```
ReferenceError: __dirname is not defined
    at loadPretrained (file:///.../node_modules/@astermind/astermind-synth/dist/astermind-synth.esm.js:2310:19)
```

### Root Cause

- `__dirname` is a **CommonJS** feature, not available in **ESM** modules
- The library file is `.esm.js` (ESM format) but uses CommonJS syntax
- Line 2310 in `astermind-synth.esm.js` uses `__dirname` to find model files

### Code Location

In `node_modules/@astermind/astermind-synth/dist/astermind-synth.esm.js` around line 2310:

```javascript
const possiblePaths = [
    path.join(__dirname, '../models/default_synth.json'), // ❌ __dirname not available in ESM
    path.join(__dirname, '../../omegasynth/models/default_synth.json'),
];
const packageRoot = findPackageRoot(__dirname) || findPackageRoot(process.cwd()); // ❌ Also uses __dirname
```

## Solution

The library needs to use the ESM equivalent of `__dirname`:

```javascript
import { fileURLToPath } from 'url';
import { dirname } from 'path';

// Get __dirname equivalent in ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Then use __dirname as normal
const possiblePaths = [
    path.join(__dirname, '../models/default_synth.json'),
    path.join(__dirname, '../../omegasynth/models/default_synth.json'),
];
```

### Fix Location

File: `src/loaders/loadPretrained.ts` (or equivalent source file)

The compiled ESM output needs to include the ESM `__dirname` workaround at the top of the file or in the function that uses it.

## Workaround (Temporary)

Until the library is fixed, you can try:

1. **Use CommonJS instead of ESM** (if possible):
   - Change `package.json` to remove `"type": "module"`
   - Use `.cjs` extensions for your files
   - Use `require()` instead of `import`

2. **Wait for library fix**: The library maintainer needs to update the code to use ESM-compatible `__dirname`.

## Testing

To verify the fix works:

```bash
npm run bootstrap
```

Should complete without the `__dirname` error.

## Related Issues

- This is separate from the audience bug fix
- The license token issue may be resolved, but this prevents testing
- Both issues need to be fixed for the library to work properly

