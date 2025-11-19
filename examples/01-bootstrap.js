// Bootstrap example: Using AsterMind Synth to generate training data for ELM
// Note: For testing, you can use julian@astermind.ai, but in production use your own email
import { loadPretrained } from '@astermind/astermind-synth';
import { ELM } from '@astermind/astermind-elm';
import { setupLicense } from '../utils/setupLicense.js';
import { config } from '../config.js';

async function bootstrapELM() {
  // Set up license token from config
  await setupLicense();
  
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
      console.log(`  Prediction: ${prediction.map(p => `${p.label} (${(p.confidence * 100).toFixed(2)}%)`).join(', ')}\n`);
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

