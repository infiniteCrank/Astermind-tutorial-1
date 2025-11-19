# Audience Bug Fix Documentation

## Problem Description

The `@astermind/astermind-synth` library has a bug in its flexible audience check that prevents valid license tokens from working when they have a different audience than expected.

### The Issue

1. **Library Configuration**: The library initializes the license runtime with `expectedAud: "astermind-synth"` (hardcoded in `initializeLicense()` function).

2. **Token Structure**: Some license tokens have `"aud": "astermind-elm"` as the audience but include `"astermind-synth"` in the `features` array:
   ```json
   {
     "aud": "astermind-elm",
     "features": ["astermind-elm", "astermind-synth"],
     ...
   }
   ```

3. **License State**: When a token with `"aud": "astermind-elm"` is validated against `expectedAud: "astermind-synth"`, the license runtime sets the state to `"invalid"` with reason `"Bad audience"`.

4. **Flexible Check Bug**: The library's `requireLicense()` function has a flexible audience check that attempts to allow tokens with different audiences if they have the required feature:
   ```javascript
   if (hasFeature("astermind-synth")) {
       // Feature is present, even if audience doesn't match - allow it
       return;
   }
   ```
   
   **However**, `hasFeature()` checks the license state first, and since the state is `"invalid"`, it returns `false` even though the feature exists in the payload.

## Current Behavior

- Token with `"aud": "astermind-elm"` and `features: ["astermind-synth"]` → ❌ **Fails**
- License state: `"invalid"` (reason: "Bad audience")
- `hasFeature("astermind-synth")` → Returns `false` (because state is invalid)
- Flexible check fails → Error thrown

## Expected Behavior

- Token with `"aud": "astermind-elm"` and `features: ["astermind-synth"]` → ✅ **Should pass**
- License state may be `"invalid"` (due to audience mismatch)
- Check payload directly for feature → Feature found
- Flexible check succeeds → License accepted

## Solution

### Fix Location

File: `src/core/license.ts` (or the compiled equivalent in `dist/`)

Function: `requireLicense()`

### Current Code (Buggy)

```typescript
function requireLicense() {
    try {
        requireFeature("astermind-synth");
    }
    catch (error) {
        const state = getLicenseState();
        // FLEXIBLE AUDIENCE CHECK: If requireFeature failed but the feature is actually present,
        // allow it (this handles tokens with different audiences that include the feature)
        if (hasFeature("astermind-synth")) {  // ❌ BUG: This checks state first, fails if invalid
            // Feature is present, even if audience doesn't match - allow it
            return;
        }
        // ... error handling
    }
}
```

### Fixed Code

```typescript
function requireLicense() {
    try {
        requireFeature("astermind-synth");
    }
    catch (error) {
        const state = getLicenseState();
        // FLEXIBLE AUDIENCE CHECK: Check payload directly instead of using hasFeature()
        // hasFeature() checks the license state first, which fails when audience doesn't match
        // We need to check the payload.features array directly to see if the feature exists
        if (state.payload?.features?.includes("astermind-synth")) {
            // Feature is present in payload, even if audience doesn't match - allow it
            return;
        }
        // Provide helpful error messages based on license state
        if (state.status === 'missing') {
            throw new Error('License token is required. Please set ASTERMIND_LICENSE_TOKEN environment variable.\n' +
                'For trial tokens, visit: https://license.astermind.ai/v1/trial/create');
        }
        else if (state.status === 'expired') {
            throw new Error(`License token has expired. Please obtain a new license token.\n` +
                `Expired at: ${state.payload?.exp ? new Date(state.payload.exp * 1000).toISOString() : 'unknown'}`);
        }
        else if (state.status === 'invalid') {
            // Only throw error if the feature is actually missing
            if (!state.payload?.features?.includes("astermind-synth")) {
                throw new Error(`License token is invalid: ${state.reason || 'unknown error'}\n` +
                    'Please verify your license token is correct.');
            }
            // If feature exists but state is invalid due to audience mismatch, we already returned above
        }
        // Re-throw original error if we can't provide better message
        throw error;
    }
}
```

### Key Changes

1. **Replace `hasFeature("astermind-synth")`** with **`state.payload?.features?.includes("astermind-synth")`**
   - This checks the payload directly instead of relying on the license state
   - Works even when state is "invalid" due to audience mismatch

2. **Update error handling** to only throw "invalid" error if the feature is actually missing
   - If the feature exists in payload, we return early (flexible check passes)
   - Only throw error if feature is truly missing

## Testing the Fix

### Test Case 1: Token with "astermind-elm" audience but "astermind-synth" feature

```javascript
// Token payload:
{
  "aud": "astermind-elm",
  "features": ["astermind-elm", "astermind-synth"]
}

// Expected: Should pass (feature exists in payload)
// Current: Fails (hasFeature() returns false due to invalid state)
// After fix: Should pass (checks payload directly)
```

### Test Case 2: Token with "astermind-synth" audience

```javascript
// Token payload:
{
  "aud": "astermind-synth",
  "features": ["astermind-synth"]
}

// Expected: Should pass (normal case)
// Current: Passes ✅
// After fix: Should still pass ✅
```

### Test Case 3: Token without "astermind-synth" feature

```javascript
// Token payload:
{
  "aud": "astermind-elm",
  "features": ["astermind-elm"]  // Missing astermind-synth
}

// Expected: Should fail (feature missing)
// Current: Fails ✅
// After fix: Should still fail ✅
```

## Implementation Steps

1. **Locate the file**: `src/core/license.ts` (or equivalent in your source structure)

2. **Find the `requireLicense()` function**

3. **Replace the flexible check**:
   ```typescript
   // OLD (buggy):
   if (hasFeature("astermind-synth")) {
       return;
   }
   
   // NEW (fixed):
   if (state.payload?.features?.includes("astermind-synth")) {
       return;
   }
   ```

4. **Update error handling** to check payload before throwing "invalid" error

5. **Rebuild and test**:
   ```bash
   npm run build
   npm test
   ```

6. **Verify with a token that has "astermind-elm" audience**:
   ```bash
   # Set token with "astermind-elm" audience
   export ASTERMIND_LICENSE_TOKEN="your-token-with-elm-audience"
   # Test that it works
   npm run bootstrap
   ```

## Why This Fix Works

- **Direct payload check**: Bypasses the license state validation that fails on audience mismatch
- **Feature-based validation**: Focuses on whether the required feature exists, not on audience matching
- **Backward compatible**: Tokens with correct audience still work normally
- **More flexible**: Allows tokens from related products (like "astermind-elm") that include the required feature

## Additional Considerations

### Alternative: Accept Multiple Audiences

Instead of (or in addition to) the flexible check, you could initialize the license runtime to accept multiple audiences:

```typescript
initLicenseRuntime({
    jwksUrl: "https://license.astermind.ai/.well-known/astermind-license-keys.json",
    expectedIss: "https://license.astermind.ai",
    expectedAud: ["astermind-synth", "astermind-elm"], // Accept multiple audiences
    jwksMaxAgeSeconds: 300,
    mode: 'strict'
});
```

However, this depends on whether the `@astermindai/license-runtime` package supports multiple audiences. The payload-based check is more reliable.

## Issue 2: Missing State Not Handled (Still Present in v0.1.4)

### Current Status

**Version 0.1.4** has improved the fix:
- ✅ Handles "invalid" state by decoding JWT payload directly
- ❌ Still doesn't handle "missing" state when token is being set asynchronously

### Problem

When `initializeLicense()` is called, it sets the token asynchronously without awaiting:

```typescript
setLicenseToken(token).catch(err => {
    console.warn("Failed to set license token from environment:", err);
});
```

This means when `requireLicense()` is called immediately after (e.g., when `loadPretrained()` creates a new `OmegaSynth`), the license state is still `"missing"` because the async `setLicenseToken()` hasn't completed yet. The current code throws an error immediately without checking if a token exists in the environment variable.

### Current Code (v0.1.4)

```typescript
function requireLicense() {
    try {
        requireFeature("astermind-synth");
    }
    catch (error) {
        const state = getLicenseState();
        
        // ✅ Handles "invalid" state - decodes JWT directly
        if (state.payload && Array.isArray(state.payload.features) && state.payload.features.includes("astermind-synth")) {
            return;
        }
        if (!state.payload && state.status === 'invalid') {
            const token = process.env.ASTERMIND_LICENSE_TOKEN;
            if (token) {
                const decodedPayload = decodeJWTPayload(token);
                if (decodedPayload && Array.isArray(decodedPayload.features) && decodedPayload.features.includes("astermind-synth")) {
                    return;
                }
            }
        }
        
        // ❌ "missing" case - throws immediately without checking env var
        if (state.status === 'missing') {
            throw new Error('License token is required...');
        }
        // ...
    }
}
```

### Additional Fix Needed

The `requireLicense()` function should handle the "missing" case by checking the environment variable and waiting for async token setting:

```typescript
if (state.status === 'missing') {
    const token = process.env.ASTERMIND_LICENSE_TOKEN;
    if (token) {
        // Token exists in env var but not set yet - wait for async setting
        // This handles the case where initializeLicense() set it asynchronously
        await new Promise(resolve => setTimeout(resolve, 100));
        const newState = getLicenseState();
        if (newState.status !== 'missing') {
            // Token was set, re-check with flexible audience check
            if (newState.payload && Array.isArray(newState.payload.features) && newState.payload.features.includes("astermind-synth")) {
                return; // Flexible check passes
            }
            // Also try decoding if payload not available
            if (!newState.payload && newState.status === 'invalid') {
                const decodedPayload = decodeJWTPayload(token);
                if (decodedPayload && Array.isArray(decodedPayload.features) && decodedPayload.features.includes("astermind-synth")) {
                    return;
                }
            }
        }
    }
    throw new Error('License token is required. Please set ASTERMIND_LICENSE_TOKEN environment variable.\n' +
        'For trial tokens, visit: https://license.astermind.ai/v1/trial/create');
}
```

**Note**: This requires making `requireLicense()` async, which might require changes to how it's called.

## Additional Issue Found: Async Token Setting (Original)

### Problem

When `initializeLicense()` is called, it sets the token asynchronously without awaiting:

```typescript
setLicenseToken(token).catch(err => {
    console.warn("Failed to set license token from environment:", err);
});
```

This means when `requireLicense()` is called immediately after (e.g., when `loadPretrained()` creates a new `OmegaSynth`), the license state is still `"missing"` because the async `setLicenseToken()` hasn't completed yet. The flexible check can't work because there's no payload to check.

### Additional Fix Needed

The `requireLicense()` function should handle the "missing" case by:

1. **Checking if token exists in environment variable** when state is "missing"
2. **Waiting briefly** for the async token setting to complete, OR
3. **Setting the token synchronously** if it's in the environment variable

Here's an updated fix:

```typescript
function requireLicense() {
    try {
        requireFeature("astermind-synth");
    }
    catch (error) {
        const state = getLicenseState();
        
        // If state is "missing", check if token is in environment variable
        // and wait for it to be set (async token setting from initializeLicense)
        if (state.status === 'missing') {
            const token = process.env.ASTERMIND_LICENSE_TOKEN;
            if (token) {
                // Token exists in env var but not set yet - wait a bit for async setting
                // This handles the case where initializeLicense() set it asynchronously
                await new Promise(resolve => setTimeout(resolve, 100));
                const newState = getLicenseState();
                if (newState.status !== 'missing') {
                    // Token was set, re-check
                    if (newState.payload?.features?.includes("astermind-synth")) {
                        return; // Flexible check passes
                    }
                }
            }
            throw new Error('License token is required. Please set ASTERMIND_LICENSE_TOKEN environment variable.\n' +
                'For trial tokens, visit: https://license.astermind.ai/v1/trial/create');
        }
        
        // FLEXIBLE AUDIENCE CHECK: Check payload directly instead of using hasFeature()
        if (state.payload?.features?.includes("astermind-synth")) {
            // Feature is present in payload, even if audience doesn't match - allow it
            return;
        }
        
        // ... rest of error handling
    }
}
```

**Note**: This requires making `requireLicense()` async, which might require changes to how it's called.

### Alternative: Make initializeLicense() Await Token Setting

A better solution might be to make `initializeLicense()` properly await the token setting:

```typescript
async function initializeLicense() {
    if (initialized) {
        return;
    }
    initLicenseRuntime({...});
    const token = process.env.ASTERMIND_LICENSE_TOKEN;
    if (token) {
        await setLicenseToken(token); // Await instead of fire-and-forget
    }
    initialized = true;
}
```

However, this requires making `initializeLicense()` async and ensuring it's awaited before `requireLicense()` is called.

## Summary

**Problem 1**: `hasFeature()` checks license state first, so it returns `false` when state is "invalid" (due to audience mismatch), even if the feature exists in the payload.

**Solution 1**: Check `state.payload?.features?.includes("astermind-synth")` directly instead of using `hasFeature()`.

**Problem 2**: `setLicenseToken()` is called asynchronously in `initializeLicense()` without awaiting, so when `requireLicense()` is called immediately, the state is "missing" with no payload.

**Solution 2**: Handle the "missing" case by checking the environment variable and waiting for async token setting, or make `initializeLicense()` await the token setting.

**Impact**: Allows tokens with different audiences (like "astermind-elm") to work as long as they include the required "astermind-synth" feature, and handles async token initialization properly.

