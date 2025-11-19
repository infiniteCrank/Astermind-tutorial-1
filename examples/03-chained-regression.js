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
 * Stage 1: ELM1 predicts intermediate features (sentiment, length, complexity)
 * Stage 2: ELM2 uses these features to predict final engagement score
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

  // Stage 1: Train ELM1 to predict intermediate features from text
  console.log('🎓 Stage 1: Training ELM1 (Text → Intermediate Features)...');
  
  // We'll train 3 separate ELMs for each feature
  const elm1Features = [];
  const elm1 = new ELM({
    categories: ['feature'],
    hiddenUnits: 128,
    maxLen: 50,
    useTokenizer: true,
    activation: 'relu',
    ridgeLambda: 1e-4
  });

  // Encode texts once
  const encodedTexts = texts.map(text => {
    const encoded = elm1.encoder.encode(text);
    return elm1.encoder.normalize(encoded);
  });

  // Train separate ELMs for each feature
  for (let featIdx = 0; featIdx < 3; featIdx++) {
    const featureELM = new ELM({
      categories: ['output'],
      hiddenUnits: 128,
      maxLen: 50,
      useTokenizer: true,
      activation: 'relu',
      ridgeLambda: 1e-4
    });

    // Convert target values to label indices (for classification-style training)
    // We'll use a regression approach by mapping values to discrete bins
    const targetValues = intermediateFeatures.map(f => f[featIdx]);
    const minVal = Math.min(...targetValues);
    const maxVal = Math.max(...targetValues);
    const numBins = 10;
    const binSize = (maxVal - minVal) / numBins;
    
    const binnedTargets = targetValues.map(val => {
      const bin = Math.min(Math.floor((val - minVal) / binSize), numBins - 1);
      return bin;
    });

    featureELM.trainFromData(encodedTexts, binnedTargets);
    elm1Features.push({ elm: featureELM, minVal, maxVal, binSize, numBins });
  }

  console.log('✅ ELM1 training complete\n');

  // Generate intermediate predictions
  const predictedIntermediate = encodedTexts.map(encoded => {
    const features = elm1Features.map(({ elm, minVal, binSize }) => {
      const pred = elm.predictFromVector(encoded, 1);
      // Convert bin prediction back to continuous value
      const bin = pred[0]?.confidence || 0;
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
  
  const elm2 = new ELM({
    categories: ['engagement'],
    hiddenUnits: 64,
    maxLen: 30,
    useTokenizer: true,
    activation: 'relu',
    ridgeLambda: 1e-4
  });

  // Convert intermediate features to text format for ELM
  const featureTexts = predictedIntermediate.map(feat => 
    feat.map(f => f.toFixed(6)).join(' ')
  );

  const encodedFeatures = featureTexts.map(text => {
    const encoded = elm2.encoder.encode(text);
    return elm2.encoder.normalize(encoded);
  });

  // Bin engagement scores for training
  const engagementValues = engagementScores.map(s => s[0]);
  const engMin = Math.min(...engagementValues);
  const engMax = Math.max(...engagementValues);
  const engBinSize = (engMax - engMin) / 10;
  const binnedEngagement = engagementValues.map(val => {
    const bin = Math.min(Math.floor((val - engMin) / engBinSize), 9);
    return bin;
  });

  elm2.trainFromData(encodedFeatures, binnedEngagement);
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
    const encoded = elm1.encoder.encode(text);
    const normalized = elm1.encoder.normalize(encoded);
    const predictedFeat = elm1Features.map(({ elm, minVal, binSize }) => {
      const pred = elm.predictFromVector(normalized, 1);
      const bin = pred[0]?.confidence || 0;
      const continuous = minVal + bin * binSize;
      return Math.max(0, Math.min(1, continuous));
    });

    // Stage 2: Intermediate Features → Engagement Score
    const featText = predictedFeat.map(f => f.toFixed(6)).join(' ');
    const encodedFeat = elm2.encoder.encode(featText);
    const normalizedFeat = elm2.encoder.normalize(encodedFeat);
    const predictedScore = elm2.predictFromVector(normalizedFeat, 1);
    const bin = predictedScore[0]?.confidence || 0;
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
    const encoded = elm1.encoder.encode(texts[i]);
    const normalized = elm1.encoder.normalize(encoded);
    const predictedFeat = elm1Features.map(({ elm, minVal, binSize }) => {
      const pred = elm.predictFromVector(normalized, 1);
      const bin = pred[0]?.confidence || 0;
      const continuous = minVal + bin * binSize;
      return Math.max(0, Math.min(1, continuous));
    });
    const featText = predictedFeat.map(f => f.toFixed(6)).join(' ');
    const encodedFeat = elm2.encoder.encode(featText);
    const normalizedFeat = elm2.encoder.normalize(encodedFeat);
    const predictedScore = elm2.predictFromVector(normalizedFeat, 1);
    const bin = predictedScore[0]?.confidence || 0;
    const finalScore = engMin + bin * engBinSize;
    totalError += Math.abs(engagementScores[i][0] - finalScore);
  }

  const mae = totalError / texts.length;
  console.log(`📊 Overall Mean Absolute Error: ${mae.toFixed(4)}`);

  return { elm1Features, elm2 };
}

if (import.meta.url === `file://${process.argv[1]}` || 
    import.meta.url.endsWith('03-chained-regression.js') ||
    process.argv[1]?.endsWith('03-chained-regression.js')) {
  createChainedRegression().catch(console.error);
}

export { createChainedRegression };

