# AsterMind ELM Tutorial

A comprehensive tutorial on using AsterMind ELM and AsterMind Synth for ensemble methods and chained regression.

## Quick Start

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Set up license token for AsterMind Synth:**
   
   **Option 1: Use config.js (Recommended)**
   
   Edit `config.js` and replace `'your-token-here'` with your actual token:
   ```javascript
   export const config = {
     licenseToken: 'your-actual-token-here',
     // ...
   };
   ```
   
   **Option 2: Use environment variable**
   ```bash
   # Get a free 30-day trial token
   curl -X POST "https://license.astermind.ai/v1/trial/create" \
     -H "Content-Type: application/json" \
     -d '{"email": "your-email@example.com", "product": "astermind-synth"}'
   
   # After verifying email, set the token
   export ASTERMIND_LICENSE_TOKEN="your-token-here"
   ```
   
   **Note:** The environment variable takes precedence if both are set.
   

3. **Check your setup:**
   ```bash
   npm run check
   ```

4. **Run examples:**
   ```bash
   # Bootstrap example
   npm run bootstrap
   
   # Ensemble classification
   npm run ensemble
   
   # Chained regression
   npm run chain
   
   # Run all examples
   npm test
   ```

## Tutorial Contents

See [TUTORIAL.md](./TUTORIAL.md) for the complete tutorial covering:

- Bootstrapping projects with AsterMind Synth
- Creating ensemble models (ELM + KELM)
- Why ensemble methods outperform standalone models
- Chaining ELMs for complex regression tasks

## Examples

- `examples/01-bootstrap.js` - Using Synth to generate training data
- `examples/02-ensemble-classification.js` - Ensemble methods for classification
- `examples/03-chained-regression.js` - Chained ELMs for regression
- `examples/04-test-all.js` - Run all examples

## Requirements

- Node.js v16 or higher
- Valid AsterMind Synth license token (free trial available)

## License

MIT

