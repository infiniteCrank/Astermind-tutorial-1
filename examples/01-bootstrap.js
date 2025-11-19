// Bootstrap example: Using AsterMind Synth to generate training data for ELM
// Note: For testing, you can use julian@astermind.ai, but in production use your own email

// CRITICAL: Set license token BEFORE importing synth library
// The library checks the environment variable at module initialization
const { setupLicense } = await import('../utils/setupLicense.js');
await setupLicense();

// Now we can safely import the synth library
const synthModule = await import('@astermind/astermind-synth');
const { loadPretrained, setLicenseTokenFromString } = synthModule;
const { ELM } = await import('@astermind/astermind-elm');
const { config } = await import('../config.js');

async function bootstrapELM() {
  // Set license token explicitly (in addition to env var)
  if (config.licenseToken && config.licenseToken !== 'your-token-here') {
    try {
      await setLicenseTokenFromString(config.licenseToken);
    } catch (error) {
      // If this fails, env var should still work
      console.warn('Note: Could not set token via function, using environment variable');
    }
  }
  
  console.log('🚀 Bootstrapping ELM with AsterMind Synth...\n');

  // Load pretrained synth model (mode from config)
  const synth = loadPretrained(config.synthMode);
  
  // Wait for initialization
  await new Promise(resolve => setTimeout(resolve, 100));

  // Generate synthetic training data
  const categories = ['first_name', 'last_name', 'email', 'phone_number'];
  const texts = [];
  const labels = [];

  console.log('📊 Generating synthetic training data...');
  for (let i = 0; i < 100; i++) {
    for (const category of categories) {
      try {
        const value = await synth.generate(category);
        texts.push(value);
        labels.push(category);
      } catch (error) {
        console.warn(`Warning: Failed to generate ${category}, skipping...`);
      }
    }
  }

  console.log(`✅ Generated ${texts.length} samples\n`);

  // Initialize ELM
  const elm = new ELM({
    categories: categories,
    hiddenUnits: 128,
    maxLen: 50,
    useTokenizer: true,
    activation: 'relu',
    weightInit: 'xavier',
    ridgeLambda: 1e-4
  });

  // Prepare training data
  const labelIndices = labels.map(l => categories.indexOf(l));
  
  // Encode texts
  const encodedTexts = texts.map(text => {
    const encoded = elm.encoder.encode(text);
    return elm.encoder.normalize(encoded);
  });

  // Train the ELM
  console.log('🎓 Training ELM model...');
  elm.trainFromData(encodedTexts, labelIndices);
  console.log('✅ Training complete!\n');

  // Test predictions
  console.log('🧪 Testing predictions:');
  const testCases = [];
  try {
    testCases.push(await synth.generate('first_name'));
    testCases.push(await synth.generate('email'));
    testCases.push(await synth.generate('phone_number'));
  } catch (error) {
    // Fallback test cases if synth fails
    testCases.push('John', 'john@example.com', '555-1234');
  }

  for (const testCase of testCases) {
    try {
      const prediction = elm.predict(testCase, 3);
      console.log(`  Input: "${testCase}"`);
      // Handle both 'confidence' and 'prob' properties, and handle NaN
      const formatted = prediction.map(p => {
        // Get confidence/prob value, handling both property names and NaN
        let conf = p.confidence ?? p.prob;
        // If value is NaN, undefined, or null, use 0
        if (conf == null || isNaN(conf) || !isFinite(conf)) {
          conf = 0;
        }
        const percent = conf > 0 ? `${(conf * 100).toFixed(2)}%` : 'N/A';
        return `${p.label} (${percent})`;
      }).join(', ');
      console.log(`  Prediction: ${formatted}\n`);
    } catch (error) {
      console.log(`  Input: "${testCase}"`);
      console.log(`  Error: ${error.message}\n`);
    }
  }

  return elm;
}

// Run if executed directly
if (import.meta.url === `file://${process.argv[1]}` || 
    import.meta.url.endsWith('01-bootstrap.js') ||
    process.argv[1]?.endsWith('01-bootstrap.js')) {
  bootstrapELM().catch(console.error);
}

export { bootstrapELM };

