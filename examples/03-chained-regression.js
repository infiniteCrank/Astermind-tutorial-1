// Chained Regression Example: Using one ELM's output as another ELM's input
// Note: For testing, you can use julian@astermind.ai, but in production use your own email

// CRITICAL: Set license token BEFORE importing synth library
const { setupLicense } = await import('../utils/setupLicense.js');
await setupLicense();

// Now we can safely import the libraries
const { ELM } = await import('@astermind/astermind-elm');
const synthModule = await import('@astermind/astermind-synth');
const { loadPretrained, setLicenseTokenFromString } = synthModule;
const { config } = await import('../config.js');

/**
 * Chained Regression Example
 * 
 * Problem: Predict user engagement score from text features
 * 
 * NOTE: This example approximates regression by discretizing continuous values
 * into bins, training classification ELMs, then mapping predicted bins back to
 * continuous values. This keeps the API purely in classification mode while
 * behaving like a coarse regressor.
 * 
 * Stage 1a: encoderELM encodes text → numeric vector (shared tokenizer-based encoder)
 * Stage 1b: featureELMs[] map encoded vector → binned features [sentiment, length, complexity]
 * Stage 2: elm2 maps predicted intermediate features → binned engagement score
 */

async function createChainedRegression() {
  // Set license token explicitly (in addition to env var)
  if (config.licenseToken && config.licenseToken !== 'your-token-here') {
    try {
      await setLicenseTokenFromString(config.licenseToken);
    } catch (error) {
      // If this fails, env var should still work
      console.warn('Note: Could not set token via function, using environment variable');
    }
  }
  
  console.log('🔗 Chained Regression Example\n');

  // Generate synthetic training data (mode from config)
  const synth = loadPretrained(config.synthMode);
  await new Promise(resolve => setTimeout(resolve, 100));

  // Generate diverse text samples
  const texts = [];
  const categories = ['first_name', 'last_name', 'email', 'company_name'];
  
  console.log('📊 Generating training data...');
  for (let i = 0; i < 500; i++) {
    try {
      const category = categories[Math.floor(Math.random() * categories.length)];
      const text = await synth.generate(category);
      texts.push(text);
    } catch (error) {
      // Fallback synthetic data
      texts.push(`Sample${i}`);
    }
  }

  // Create synthetic intermediate features (what ELM1 should learn to predict)
  // These represent: [sentiment_score, length_score, complexity_score]
  const intermediateFeatures = texts.map(text => {
    const sentiment = Math.sin(text.length * 0.1) * 0.5 + 0.5; // 0-1
    const length = Math.min(text.length / 50, 1.0); // normalized length
    const complexity = (text.match(/[A-Z]/g) || []).length / Math.max(text.length, 1); // capital ratio
    return [sentiment, length, complexity];
  });

  // Create final target (what ELM2 should predict)
  // Engagement score = f(sentiment, length, complexity) + noise
  const engagementScores = intermediateFeatures.map(([sent, len, comp]) => {
    // Complex non-linear relationship
    const base = sent * 0.4 + len * 0.3 + comp * 0.3;
    const interaction = sent * len * 0.2; // interaction term
    const noise = (Math.random() - 0.5) * 0.1; // small noise
    return [base + interaction + noise];
  });

  console.log(`✅ Generated ${texts.length} training samples`);
  console.log(`  Intermediate features: ${intermediateFeatures[0].length} per sample`);
  console.log(`  Target scores: ${engagementScores.length}\n`);

  // Stage 1a: Create encoder ELM for text → numeric vector conversion
  console.log('🎓 Stage 1a: Building text encoder...');
  
  // encoderELM: text → numeric vector (shared tokenizer-based encoder)
  const encoderELM = new ELM({
    categories: ['feature'],
    hiddenUnits: 128,
    maxLen: 50,
    useTokenizer: true,
    activation: 'relu',
    ridgeLambda: 1e-4
  });

  // Extract and guard encoder
  const encoder = encoderELM.encoder;
  if (!encoder) {
    throw new Error('Encoder not initialized for encoderELM');
  }

  // Encode texts once using the shared encoder
  const encodedTexts = texts.map(text => {
    const encoded = encoder.encode(text);
    return encoder.normalize(encoded);
  });

  console.log(`✅ Encoder ready (vector dimension: ${encodedTexts[0].length})\n`);

  // Stage 1b: Train feature ELMs to predict intermediate features from encoded vectors
  console.log('🎓 Stage 1b: Training feature ELMs (Encoded Vector → Binned Features)...');
  
  // featureELMs[i]: encoded vector → binned feature i (discretized regression)
  const featureELMs = [];

  // Train separate ELMs for each feature (using vector-based training)
  for (let featIdx = 0; featIdx < 3; featIdx++) {
    // NOTE: We approximate regression by:
    // 1) Binning continuous values into numBins classes,
    // 2) Training a classification ELM,
    // 3) Mapping the predicted bin back to a continuous score.
    // This keeps the API purely in classification mode while behaving like a coarse regressor.
    
    const targetValues = intermediateFeatures.map(f => f[featIdx]);
    const minVal = Math.min(...targetValues);
    const maxVal = Math.max(...targetValues);
    const numBins = 10;
    
    // Guard against division by zero (degenerate case: all values equal)
    let binSize = (maxVal - minVal) / numBins;
    if (!isFinite(binSize) || binSize === 0) {
      // Degenerate case: all values equal; just map everything to bin 0
      binSize = 1;
    }
    
    const binnedTargets = targetValues.map(val => {
      if (!isFinite(binSize) || binSize === 0) return 0;
      const bin = Math.min(Math.floor((val - minVal) / binSize), numBins - 1);
      return Math.max(0, bin); // Ensure non-negative
    });

    // Create ELM for vector-based training (useTokenizer: false)
    // Vector-mode ELMs use inputSize, not maxLen
    const featureELM = new ELM({
      useTokenizer: false,
      inputSize: encodedTexts[0].length,
      categories: Array.from({ length: numBins }, (_, i) => `bin${i}`),
      hiddenUnits: 128,
      activation: 'relu',
      ridgeLambda: 1e-4
    });

    featureELM.trainFromData(encodedTexts, binnedTargets);
    featureELMs.push({ elm: featureELM, minVal, maxVal, binSize, numBins });
  }

  console.log('✅ Feature ELMs training complete\n');

  // Generate intermediate predictions (used for Stage 2 training)
  const predictedIntermediate = encodedTexts.map(encoded => {
    const features = featureELMs.map(({ elm, minVal, binSize }) => {
      // predictFromVector expects array of vectors and returns array of arrays
      const predArray = elm.predictFromVector([encoded], 1);
      const pred = predArray[0] && predArray[0][0] ? predArray[0][0] : null;
      // Get bin index from label (e.g., "bin5" -> 5)
      const binLabel = pred?.label || 'bin0';
      const bin = parseInt(binLabel.replace('bin', '')) || 0;
      const continuous = minVal + bin * binSize;
      return Math.max(0, Math.min(1, continuous)); // clamp to [0, 1]
    });
    return features;
  });

  console.log('📈 Sample intermediate predictions:');
  console.log(`  Actual:    [${intermediateFeatures[0].map(f => f.toFixed(3)).join(', ')}]`);
  console.log(`  Predicted: [${predictedIntermediate[0].map(f => f.toFixed(3)).join(', ')}]\n`);

  // Stage 2: Train ELM2 to predict engagement from intermediate features
  console.log('🎓 Stage 2: Training ELM2 (Intermediate Features → Engagement Score)...');
  
  // Use predicted intermediate features for training ELM2,
  // so Stage 2 reflects real chained performance (not an oracle using ground-truth features).
  // Since we have 3 features, we can use them directly as a 3D vector
  const featureVectors = predictedIntermediate.map(feat => feat); // Already vectors [sentiment, length, complexity]

  // Bin engagement scores for training
  // NOTE: We approximate regression by binning continuous engagement scores into discrete classes
  const engagementValues = engagementScores.map(s => s[0]);
  const engMin = Math.min(...engagementValues);
  const engMax = Math.max(...engagementValues);
  const numEngBins = 10;
  
  // Guard against division by zero (degenerate case: all values equal)
  let engBinSize = (engMax - engMin) / numEngBins;
  if (!isFinite(engBinSize) || engBinSize === 0) {
    // Degenerate case: all values equal; just map everything to bin 0
    engBinSize = 1;
  }
  
  const binnedEngagement = engagementValues.map(val => {
    if (!isFinite(engBinSize) || engBinSize === 0) return 0;
    const bin = Math.min(Math.floor((val - engMin) / engBinSize), numEngBins - 1);
    return Math.max(0, bin); // Ensure non-negative
  });

  // ELM2: classification over engagement bins (discretized regression target)
  // Vector-mode ELMs use inputSize, not maxLen
  const elm2 = new ELM({
    useTokenizer: false,
    inputSize: 3, // sentiment, length, complexity
    categories: Array.from({ length: numEngBins }, (_, i) => `bin${i}`),
    hiddenUnits: 64,
    activation: 'relu',
    ridgeLambda: 1e-4
  });

  elm2.trainFromData(featureVectors, binnedEngagement);
  console.log('✅ ELM2 training complete\n');

  // Test the chained model
  console.log('🧪 Testing Chained Model:\n');
  
  const testSize = Math.min(5, texts.length);
  const testTexts = texts.slice(0, testSize);
  const testActualIntermediate = intermediateFeatures.slice(0, testSize);
  const testActualScores = engagementScores.slice(0, testSize);

  for (let i = 0; i < testTexts.length; i++) {
    const text = testTexts[i];
    const actualIntermediate = testActualIntermediate[i];
    const actualScore = testActualScores[i][0];

    // Stage 1: Text → Intermediate Features
    const encoded = encoder.encode(text);
    const normalized = encoder.normalize(encoded);
    const predictedFeat = featureELMs.map(({ elm, minVal, binSize }) => {
      const predArray = elm.predictFromVector([normalized], 1);
      const pred = predArray[0] && predArray[0][0] ? predArray[0][0] : null;
      const binLabel = pred?.label || 'bin0';
      const bin = parseInt(binLabel.replace('bin', '')) || 0;
      const continuous = minVal + bin * binSize;
      return Math.max(0, Math.min(1, continuous));
    });

    // Stage 2: Intermediate Features → Engagement Score (use features directly as vector)
    const predictedScoreArray = elm2.predictFromVector([predictedFeat], 1);
    const predictedScore = predictedScoreArray[0] && predictedScoreArray[0][0] ? predictedScoreArray[0][0] : null;
    const binLabel = predictedScore?.label || 'bin0';
    const bin = parseInt(binLabel.replace('bin', '')) || 0;
    const finalScore = engMin + bin * engBinSize;

    console.log(`Text: "${text}"`);
    console.log(`  Intermediate Features:`);
    console.log(`    Actual:    [${actualIntermediate.map(f => f.toFixed(3)).join(', ')}]`);
    console.log(`    Predicted: [${predictedFeat.map(f => f.toFixed(3)).join(', ')}]`);
    console.log(`  Engagement Score:`);
    console.log(`    Actual:    ${actualScore.toFixed(4)}`);
    console.log(`    Predicted: ${finalScore.toFixed(4)}`);
    console.log(`    Error:     ${Math.abs(actualScore - finalScore).toFixed(4)}\n`);
  }

  // Calculate overall error
  let totalError = 0;
  for (let i = 0; i < texts.length; i++) {
    const encoded = encoder.encode(texts[i]);
    const normalized = encoder.normalize(encoded);
    const predictedFeat = featureELMs.map(({ elm, minVal, binSize }) => {
      const predArray = elm.predictFromVector([normalized], 1);
      const pred = predArray[0] && predArray[0][0] ? predArray[0][0] : null;
      const binLabel = pred?.label || 'bin0';
      const bin = parseInt(binLabel.replace('bin', '')) || 0;
      const continuous = minVal + bin * binSize;
      return Math.max(0, Math.min(1, continuous));
    });
    // Use features directly as vector (no text encoding needed)
    const predictedScoreArray = elm2.predictFromVector([predictedFeat], 1);
    const predictedScore = predictedScoreArray[0] && predictedScoreArray[0][0] ? predictedScoreArray[0][0] : null;
    const binLabel = predictedScore?.label || 'bin0';
    const bin = parseInt(binLabel.replace('bin', '')) || 0;
    const finalScore = engMin + bin * engBinSize;
    totalError += Math.abs(engagementScores[i][0] - finalScore);
  }

  const mae = totalError / texts.length;
  console.log(`📊 Overall Mean Absolute Error: ${mae.toFixed(4)}`);

  return { featureELMs, elm2 };
}

if (import.meta.url === `file://${process.argv[1]}` || 
    import.meta.url.endsWith('03-chained-regression.js') ||
    process.argv[1]?.endsWith('03-chained-regression.js')) {
  createChainedRegression().catch(console.error);
}

export { createChainedRegression };

