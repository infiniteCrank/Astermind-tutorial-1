// Ensemble Classification Example: Combining ELM and KELM models
// Note: For testing, you can use julian@astermind.ai, but in production use your own email

// CRITICAL: Set license token BEFORE importing synth library
const { setupLicense } = await import('../utils/setupLicense.js');
await setupLicense();

// Now we can safely import the libraries
const { ELM, KernelELM } = await import('@astermind/astermind-elm');
const synthModule = await import('@astermind/astermind-synth');
const { loadPretrained, setLicenseTokenFromString } = synthModule;
const { config } = await import('../config.js');

/**
 * Ensemble model structure matching the official example
 */
class EnsembleModel {
  constructor(elm, kelm, encoder, uniqueLabels) {
    this.elm = elm;
    this.kelm = kelm;
    this.encoder = encoder;
    this.uniqueLabels = uniqueLabels;
  }

  /**
   * Get ensemble prediction from a pre-encoded vector using full probability fusion.
   * This matches the official example implementation.
   * 
   * Uses:
   * - elm.predictFromVector([x], ...) to get per-class probs
   * - kelm.predictProbaFromVectors([x]) to get per-class probs
   * Then fuses them and takes topK AFTER fusion.
   */
  predictFromVector(x, topK = 3, kelmWeight = 0.6) {
    const elmWeight = 1 - kelmWeight;

    // ELM probabilities from vector - get ALL labels, not just topK
    const elmProbsArr = this.elm.predictFromVector([x], this.uniqueLabels.length)[0];
    
    // Convert ELM predictions to probability array indexed by label index
    const elmProbs = new Array(this.uniqueLabels.length).fill(0);
    for (const p of elmProbsArr) {
      const idx = this.uniqueLabels.indexOf(p.label);
      if (idx >= 0) elmProbs[idx] = p.prob || p.confidence || 0;
    }

    // KernelELM probabilities from vector
    const kelmProbs = this.kelm.predictProbaFromVectors([x])[0];

    // Fuse probabilities: weighted combination
    const combined = [];
    let sum = 0;
    for (let i = 0; i < this.uniqueLabels.length; i++) {
      const p = elmWeight * elmProbs[i] + kelmWeight * kelmProbs[i];
      combined.push({ label: this.uniqueLabels[i], prob: p });
      sum += p;
    }

    // Normalize to ensure probabilities sum to 1
    if (sum > 0) {
      for (const c of combined) c.prob /= sum;
    }

    // Sort and take topK
    combined.sort((a, b) => b.prob - a.prob);
    return combined.slice(0, topK);
  }

  /**
   * Predict from text input (convenience method)
   */
  predict(text, topK = 3, kelmWeight = 0.6) {
    // Encode text to vector
    const encoded = this.encoder.encode(text);
    const normalized = this.encoder.normalize(encoded);
    
    // Use vector-based prediction
    return this.predictFromVector(normalized, topK, kelmWeight);
  }
}

async function runEnsembleExample() {
  // Set license token explicitly (in addition to env var)
  if (config.licenseToken && config.licenseToken !== 'your-token-here') {
    try {
      await setLicenseTokenFromString(config.licenseToken);
    } catch (error) {
      // If this fails, env var should still work
      console.warn('Note: Could not set token via function, using environment variable');
    }
  }
  
  console.log('🎯 Ensemble Classification Example\n');

  // Generate training data (mode from config)
  const synth = loadPretrained(config.synthMode);
  await new Promise(resolve => setTimeout(resolve, 100));

  const categories = ['first_name', 'last_name', 'email', 'phone_number'];
  const texts = [];
  const labels = [];

  console.log('📊 Generating training data...');
  for (let i = 0; i < 200; i++) {
    for (const category of categories) {
      try {
        const value = await synth.generate(category);
        texts.push(value);
        labels.push(category);
      } catch (error) {
        // Fallback: create synthetic data
        if (category === 'first_name') texts.push(`Name${i}`);
        else if (category === 'last_name') texts.push(`Surname${i}`);
        else if (category === 'email') texts.push(`user${i}@example.com`);
        else if (category === 'phone_number') texts.push(`555-${1000 + i}`);
        labels.push(category);
      }
    }
  }

  console.log(`✅ Generated ${texts.length} training samples\n`);

  // Prepare data
  const labelIndices = labels.map(l => categories.indexOf(l));

  // Train ELM
  console.log('🎓 Training ELM...');
  const elm = new ELM({
    categories: categories,
    hiddenUnits: 128,
    maxLen: 50,
    useTokenizer: true,
    activation: 'relu',
    ridgeLambda: 1e-4
  });

  const encodedTexts = texts.map(text => {
    const encoded = elm.encoder.encode(text);
    return elm.encoder.normalize(encoded);
  });

  elm.trainFromData(encodedTexts, labelIndices);
  console.log('✅ ELM training complete\n');

  // Train KELM (using vectorized inputs)
  // Using improved gamma calculation matching the official example
  console.log('🎓 Training KELM...');
  const inputDim = encodedTexts[0].length;
  const gammaMultiplier = 0.05; // Tuned for high-dimensional text data
  const gamma = Math.max(1e-6, gammaMultiplier / Math.sqrt(inputDim)); // sqrt scaling for better balance
  
  const baseLandmarks = Math.floor(Math.sqrt(encodedTexts.length));
  const nystromM = Math.min(2000, Math.floor(baseLandmarks * 3)); // 3x sqrt(N) for better approximation
  
  const kelm = new KernelELM({
    outputDim: categories.length,
    kernel: { type: 'rbf', gamma: gamma },
    mode: 'nystrom',
    nystrom: { 
      m: nystromM, 
      strategy: 'uniform', 
      whiten: true // Whitening helps with numerical stability
    },
    ridgeLambda: 0.001 // Moderate regularization
  });
  
  console.log(`  Input dimension: ${inputDim}`);
  console.log(`  Gamma: ${gamma.toExponential(3)} (sqrt scaling)`);
  console.log(`  Nyström landmarks: ${nystromM}`);

  // Convert labels to one-hot
  const oneHotLabels = labelIndices.map(idx => {
    const oneHot = new Array(categories.length).fill(0);
    oneHot[idx] = 1;
    return oneHot;
  });

  kelm.fit(encodedTexts, oneHotLabels);
  console.log('✅ KELM training complete\n');

  // Create ensemble using the proper structure (matching official example)
  console.log('🔗 Creating ensemble...');
  const ensemble = new EnsembleModel(elm, kelm, elm.encoder, categories);
  console.log('✅ Ensemble created\n');

  // Test individual models vs ensemble
  console.log('🧪 Testing Models:\n');
  const testCases = [];
  try {
    testCases.push(await synth.generate('first_name'));
    testCases.push(await synth.generate('email'));
    testCases.push(await synth.generate('phone_number'));
  } catch (error) {
    testCases.push('John', 'john@example.com', '555-1234');
  }

  for (const testCase of testCases) {
    console.log(`Input: "${testCase}"`);
    
    try {
      const elmPred = elm.predict(testCase, 1)[0];
      const elmConf = elmPred.confidence ?? elmPred.prob ?? 0;
      const elmPercent = (elmConf != null && !isNaN(elmConf) && isFinite(elmConf)) 
        ? `${(elmConf * 100).toFixed(2)}%` 
        : 'N/A';
      console.log(`  ELM:        ${elmPred.label} (${elmPercent})`);
    } catch (error) {
      console.log(`  ELM:        Error - ${error.message}`);
    }
    
    try {
      // For KELM, we need to encode and predict
      const encoded = elm.encoder.encode(testCase);
      const normalized = elm.encoder.normalize(encoded);
      const kelmProbs = kelm.predictProbaFromVectors([normalized])[0];
      const kelmIdx = kelmProbs.indexOf(Math.max(...kelmProbs));
      const kelmConf = kelmProbs[kelmIdx];
      console.log(`  KELM:       ${categories[kelmIdx]} (${(kelmConf * 100).toFixed(2)}%)`);
    } catch (error) {
      console.log(`  KELM:       Error - ${error.message}`);
    }
    
    try {
      const ensemblePred = ensemble.predict(testCase, 1)[0];
      console.log(`  Ensemble:   ${ensemblePred.label} (${(ensemblePred.prob * 100).toFixed(2)}%)`);
    } catch (error) {
      console.log(`  Ensemble:   Error - ${error.message}`);
    }
    console.log('');
  }

  return { elm, kelm, ensemble };
}

if (import.meta.url === `file://${process.argv[1]}` || 
    import.meta.url.endsWith('02-ensemble-classification.js') ||
    process.argv[1]?.endsWith('02-ensemble-classification.js')) {
  runEnsembleExample().catch(console.error);
}

export { EnsembleModel, runEnsembleExample };

