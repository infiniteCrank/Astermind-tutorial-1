// Ensemble Classification Example: Combining ELM and KELM models
// Note: For testing, you can use julian@astermind.ai, but in production use your own email

// CRITICAL: Set license token BEFORE importing synth library
const { setupLicense } = await import('../utils/setupLicense.js');
await setupLicense();

// Now we can safely import the libraries
const { ELM, KernelELM } = await import('@astermind/astermind-elm');
const synthModule = await import('@astermind/astermind-synth');
const { loadPretrained } = synthModule;
const { config } = await import('../config.js');

/**
 * Ensemble model structure matching the canonical pattern
 * @typedef {Object} EnsembleModel
 * @property {ELM} elm
 * @property {KernelELM} kelm
 * @property {any} encoder - Shared encoder instance
 * @property {string[]} uniqueLabels
 */

/**
 * Build a shared encoder for text-to-vector conversion
 * @param {string[]} uniqueLabels - Array of unique label strings
 * @returns {any} Encoder instance with encode() and normalize() methods
 */
function buildSharedEncoder(uniqueLabels) {
  const elm = new ELM({
    categories: uniqueLabels,
    hiddenUnits: 128,
    maxLen: 50,
    useTokenizer: true,
    activation: 'relu',
    ridgeLambda: 1e-4
  });
  
  // Set categories if method is available
  if (elm.setCategories) {
    elm.setCategories(uniqueLabels);
  }
  
  return elm.encoder;
}

/**
 * Train ELM from pre-encoded vectors
 * @param {number[][]} X - Encoded feature vectors
 * @param {string[]} labels - Label strings for each sample
 * @param {string[]} uniqueLabels - Unique label set
 * @param {Object} config - Training config
 * @param {number} config.hiddenUnits - Number of hidden units
 * @param {string} config.activation - Activation function
 * @param {number} config.ridgeLambda - Ridge regularization parameter
 * @returns {ELM} Trained ELM model
 */
function trainELMFromVectors(X, labels, uniqueLabels, config = {}) {
  const {
    hiddenUnits = 128,
    activation = 'relu',
    ridgeLambda = 1e-4
  } = config;
  
  // Create ELM with categories set up properly
  const elm = new ELM({
    categories: uniqueLabels,
    hiddenUnits: hiddenUnits,
    maxLen: X[0].length, // Use maxLen instead of inputSize
    useTokenizer: false,
    activation: activation,
    ridgeLambda: ridgeLambda
  });
  
  // Ensure categories are set
  if (elm.setCategories) {
    elm.setCategories(uniqueLabels);
  }
  
  const labelIndices = labels.map(l => uniqueLabels.indexOf(l));
  elm.trainFromData(X, labelIndices);
  
  return elm;
}

/**
 * Train KernelELM from pre-encoded vectors
 * @param {number[][]} X - Encoded feature vectors
 * @param {string[]} labels - Label strings for each sample
 * @param {string[]} uniqueLabels - Unique label set
 * @param {Object} config - Training config
 * @param {string} config.kernelType - Kernel type ('rbf', 'linear', 'poly')
 * @param {number} config.ridgeLambda - Ridge regularization parameter
 * @param {number} config.gammaMultiplier - Multiplier for data-driven gamma
 * @param {number} config.nystromMultiplier - Multiplier for Nyström landmarks
 * @returns {KernelELM} Trained KernelELM model
 */
function trainKernelELMFromVectors(X, labels, uniqueLabels, config = {}) {
  const {
    kernelType = 'rbf',
    ridgeLambda = 0.001,
    gammaMultiplier = 0.05,
    nystromMultiplier = 3
  } = config;
  
  // Convert labels to one-hot encoding
  const oneHotLabels = labels.map(label => {
    const idx = uniqueLabels.indexOf(label);
    const oneHot = new Array(uniqueLabels.length).fill(0);
    oneHot[idx] = 1;
    return oneHot;
  });
  
  // Data-driven RBF gamma calculation (median squared distances)
  let gamma = 1.0 / X[0].length; // Default fallback
  if (kernelType === 'rbf' && X.length > 1) {
    const distances = [];
    const sampleSize = Math.min(100, X.length); // Sample for efficiency
    
    for (let i = 0; i < sampleSize; i++) {
      for (let j = i + 1; j < sampleSize; j++) {
        let distSq = 0;
        for (let k = 0; k < X[i].length; k++) {
          const diff = X[i][k] - X[j][k];
          distSq += diff * diff;
        }
        distances.push(distSq);
      }
    }
    
    if (distances.length > 0) {
      distances.sort((a, b) => a - b);
      const medianDistSq = distances[Math.floor(distances.length / 2)];
      gamma = Math.max(1e-6, gammaMultiplier / Math.sqrt(medianDistSq || 1));
    }
  }
  
  // Nyström landmarks calculation
  const N = X.length;
  const baseLandmarks = Math.floor(Math.sqrt(N));
  const m = Math.min(2000, Math.floor(baseLandmarks * nystromMultiplier));
  
  const kelm = new KernelELM({
    outputDim: uniqueLabels.length,
    kernel: { type: kernelType, gamma: gamma },
    mode: 'nystrom',
    nystrom: {
      m: m,
      strategy: 'random',
      whiten: true
    },
    ridgeLambda: ridgeLambda
  });
  
  kelm.fit(X, oneHotLabels);
  
  return kelm;
}

/**
 * Get ensemble prediction from a pre-encoded vector
 * @param {EnsembleModel} ensemble - Ensemble model structure
 * @param {number[]} x - Encoded feature vector
 * @param {number} topK - Number of top predictions to return
 * @param {number} kelmWeight - Weight for KELM in fusion (default 0.6)
 * @returns {Array<{label: string, prob: number}>} Top K predictions with probabilities
 */
function getEnsemblePredictionFromVector(ensemble, x, topK = 3, kelmWeight = 0.6) {
  const elmWeight = 1 - kelmWeight;
  
  // Get ELM probabilities for all labels
  // predictFromVector returns array of arrays: [[{label, prob}, ...]]
  const elmProbsArr = ensemble.elm.predictFromVector([x], ensemble.uniqueLabels.length)[0] || [];
  const elmProbs = new Array(ensemble.uniqueLabels.length).fill(0);
  for (const p of elmProbsArr) {
    const idx = ensemble.uniqueLabels.indexOf(p.label);
    if (idx >= 0) {
      elmProbs[idx] = p.prob || p.confidence || 0;
    }
  }
  
  // Get KELM probabilities
  const kelmProbs = ensemble.kelm.predictProbaFromVectors([x])[0];
  
  // Fuse probabilities: weighted combination
  const combined = [];
  let sum = 0;
  for (let i = 0; i < ensemble.uniqueLabels.length; i++) {
    const p = elmWeight * elmProbs[i] + kelmWeight * kelmProbs[i];
    combined.push({ label: ensemble.uniqueLabels[i], prob: p });
    sum += p;
  }
  
  // Normalize to ensure probabilities sum to 1
  if (sum > 0) {
    for (const c of combined) {
      c.prob /= sum;
    }
  }
  
  // Sort and take topK
  combined.sort((a, b) => b.prob - a.prob);
  return combined.slice(0, topK);
}

/**
 * Predict from text input (convenience helper)
 * @param {EnsembleModel} ensemble - Ensemble model structure
 * @param {any} encoder - Shared encoder instance
 * @param {string} text - Input text
 * @param {number} topK - Number of top predictions to return
 * @param {number} kelmWeight - Weight for KELM in fusion
 * @returns {Array<{label: string, prob: number}>} Top K predictions
 */
function predictText(ensemble, encoder, text, topK = 3, kelmWeight = 0.6) {
  const encoded = encoder.encode(text);
  const normalized = encoder.normalize(encoded);
  return getEnsemblePredictionFromVector(ensemble, normalized, topK, kelmWeight);
}

/**
 * Test ELM and collect accuracy results from vectors
 * @param {ELM} elm - Trained ELM model
 * @param {number[][]} X - Test feature vectors
 * @param {string[]} labels - True labels
 * @param {string[]} uniqueLabels - Unique label set
 * @returns {Object} Results with accuracy and per-label accuracy
 */
function testELMAndCollectResultsVector(elm, X, labels, uniqueLabels) {
  let correct = 0;
  let total = 0;
  const perLabel = {};
  
  for (let i = 0; i < uniqueLabels.length; i++) {
    perLabel[uniqueLabels[i]] = { correct: 0, total: 0 };
  }
  
  for (let i = 0; i < X.length; i++) {
    try {
      const predArray = elm.predictFromVector([X[i]], uniqueLabels.length);
      // predictFromVector returns array of arrays: [[{label, prob}, ...]]
      const pred = predArray[0] && predArray[0][0] ? predArray[0][0] : null;
      const predictedLabel = pred?.label || null;
      const trueLabel = labels[i];
      
      total++;
      perLabel[trueLabel].total++;
      
      if (predictedLabel && predictedLabel === trueLabel) {
        correct++;
        perLabel[trueLabel].correct++;
      }
    } catch (error) {
      // Skip on error, but count as incorrect
      total++;
      perLabel[labels[i]].total++;
    }
  }
  
  const accuracy = total > 0 ? correct / total : 0;
  
  return {
    correct,
    total,
    accuracy,
    perLabelAccuracy: Object.fromEntries(
      Object.entries(perLabel).map(([label, stats]) => [
        label,
        stats.total > 0 ? stats.correct / stats.total : 0
      ])
    )
  };
}

/**
 * Test KernelELM and collect accuracy results from vectors
 * @param {KernelELM} kelm - Trained KernelELM model
 * @param {number[][]} X - Test feature vectors
 * @param {string[]} labels - True labels
 * @param {string[]} uniqueLabels - Unique label set
 * @returns {Object} Results with accuracy and per-label accuracy
 */
function testKELMAndCollectResultsVector(kelm, X, labels, uniqueLabels) {
  let correct = 0;
  let total = 0;
  const perLabel = {};
  
  for (let i = 0; i < uniqueLabels.length; i++) {
    perLabel[uniqueLabels[i]] = { correct: 0, total: 0 };
  }
  
  for (let i = 0; i < X.length; i++) {
    const probs = kelm.predictProbaFromVectors([X[i]])[0];
    const predictedIdx = probs.indexOf(Math.max(...probs));
    const predictedLabel = uniqueLabels[predictedIdx];
    const trueLabel = labels[i];
    
    total++;
    perLabel[trueLabel].total++;
    
    if (predictedLabel === trueLabel) {
      correct++;
      perLabel[trueLabel].correct++;
    }
  }
  
  const accuracy = total > 0 ? correct / total : 0;
  
  return {
    correct,
    total,
    accuracy,
    perLabelAccuracy: Object.fromEntries(
      Object.entries(perLabel).map(([label, stats]) => [
        label,
        stats.total > 0 ? stats.correct / stats.total : 0
      ])
    )
  };
}

/**
 * Test Ensemble and collect accuracy results from vectors
 * @param {EnsembleModel} ensemble - Ensemble model structure
 * @param {number[][]} X - Test feature vectors
 * @param {string[]} labels - True labels
 * @param {string[]} uniqueLabels - Unique label set
 * @param {number} kelmWeight - Weight for KELM in fusion
 * @returns {Object} Results with accuracy and per-label accuracy
 */
function testEnsembleAndCollectResultsVector(ensemble, X, labels, uniqueLabels, kelmWeight = 0.6) {
  let correct = 0;
  let total = 0;
  const perLabel = {};
  
  for (let i = 0; i < uniqueLabels.length; i++) {
    perLabel[uniqueLabels[i]] = { correct: 0, total: 0 };
  }
  
  for (let i = 0; i < X.length; i++) {
    const pred = getEnsemblePredictionFromVector(ensemble, X[i], 1, kelmWeight)[0];
    const predictedLabel = pred.label;
    const trueLabel = labels[i];
    
    total++;
    perLabel[trueLabel].total++;
    
    if (predictedLabel === trueLabel) {
      correct++;
      perLabel[trueLabel].correct++;
    }
  }
  
  const accuracy = total > 0 ? correct / total : 0;
  
  return {
    correct,
    total,
    accuracy,
    perLabelAccuracy: Object.fromEntries(
      Object.entries(perLabel).map(([label, stats]) => [
        label,
        stats.total > 0 ? stats.correct / stats.total : 0
      ])
    )
  };
}

/**
 * Split data into train/test sets with balanced labels
 * @param {string[]} texts - Input texts
 * @param {string[]} labels - Labels
 * @param {number} testRatio - Ratio of test data (default 0.2)
 * @returns {Object} Train and test splits
 */
function splitTrainTest(texts, labels, testRatio = 0.2) {
  const uniqueLabels = [...new Set(labels)];
  const trainTexts = [];
  const trainLabels = [];
  const testTexts = [];
  const testLabels = [];
  
  // Split per label to maintain balance
  for (const label of uniqueLabels) {
    const indices = labels.map((l, idx) => l === label ? idx : -1).filter(idx => idx >= 0);
    const testCount = Math.floor(indices.length * testRatio);
    const shuffled = [...indices].sort(() => Math.random() - 0.5);
    
    for (let i = 0; i < shuffled.length; i++) {
      const idx = shuffled[i];
      if (i < testCount) {
        testTexts.push(texts[idx]);
        testLabels.push(labels[idx]);
      } else {
        trainTexts.push(texts[idx]);
        trainLabels.push(labels[idx]);
      }
    }
  }
  
  return { trainTexts, trainLabels, testTexts, testLabels };
}

async function runEnsembleExample() {
  console.log('🎯 Ensemble Classification Example\n');

  // Generate training data (mode from config)
  const synth = loadPretrained(config.synthMode);
  await new Promise(resolve => setTimeout(resolve, 100));

  const uniqueLabels = ['first_name', 'last_name', 'email', 'phone_number'];
  const texts = [];
  const labels = [];

  console.log('📊 Generating synthetic training data...');
  for (let i = 0; i < 200; i++) {
    for (const category of uniqueLabels) {
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

  console.log(`✅ Generated ${texts.length} samples\n`);

  // Split into train/test sets (80/20 per label)
  console.log('📊 Splitting data into train/test sets...');
  const { trainTexts, trainLabels, testTexts, testLabels } = splitTrainTest(texts, labels, 0.2);
  console.log(`  Train: ${trainTexts.length} samples`);
  console.log(`  Test:  ${testTexts.length} samples\n`);

  // Build shared encoder
  console.log('🔧 Building shared encoder...');
  const encoder = buildSharedEncoder(uniqueLabels);
  console.log('✅ Encoder ready\n');

  // Pre-encode all texts
  console.log('🔄 Encoding texts to vectors...');
  const encodedTrain = trainTexts.map(text => {
    const encoded = encoder.encode(text);
    return encoder.normalize(encoded);
  });
  const encodedTest = testTexts.map(text => {
    const encoded = encoder.encode(text);
    return encoder.normalize(encoded);
  });
  console.log(`  Input dimension: ${encodedTrain[0].length}\n`);

  // Train ELM from vectors
  console.log('🎓 Training ELM from vectors...');
  const elm = trainELMFromVectors(encodedTrain, trainLabels, uniqueLabels, {
    hiddenUnits: 128,
    activation: 'relu',
    ridgeLambda: 1e-4
  });
  console.log('✅ ELM training complete\n');

  // Train KernelELM from vectors
  console.log('🎓 Training KernelELM from vectors...');
  const kelm = trainKernelELMFromVectors(encodedTrain, trainLabels, uniqueLabels, {
    kernelType: 'rbf',
    ridgeLambda: 0.001,
    gammaMultiplier: 0.05,
    nystromMultiplier: 3
  });
  console.log('✅ KernelELM training complete\n');

  // Create ensemble
  console.log('🔗 Creating ensemble...');
  const ensemble = {
    elm: elm,
    kelm: kelm,
    encoder: encoder,
    uniqueLabels: uniqueLabels
  };
  console.log('✅ Ensemble created\n');

  // Test models and collect results
  console.log('🧪 Testing models on held-out test set...\n');
  
  const elmResults = testELMAndCollectResultsVector(elm, encodedTest, testLabels, uniqueLabels);
  const kelmResults = testKELMAndCollectResultsVector(kelm, encodedTest, testLabels, uniqueLabels);
  const ensembleResults = testEnsembleAndCollectResultsVector(ensemble, encodedTest, testLabels, uniqueLabels, 0.6);

  // Print comparison report
  console.log('========================');
  console.log('Model Comparison (Form Fields)');
  console.log('========================');
  console.log(`ELM:        ${elmResults.correct}/${elmResults.total} (${(elmResults.accuracy * 100).toFixed(2)}%)`);
  console.log(`KernelELM:  ${kelmResults.correct}/${kelmResults.total} (${(kelmResults.accuracy * 100).toFixed(2)}%)`);
  console.log(`Ensemble:   ${ensembleResults.correct}/${ensembleResults.total} (${(ensembleResults.accuracy * 100).toFixed(2)}%)`);
  console.log('========================\n');

  // Show sample predictions
  console.log('📝 Sample Predictions:\n');
  const sampleIndices = [0, Math.floor(testTexts.length / 2), testTexts.length - 1].slice(0, 3);
  
  for (const idx of sampleIndices) {
    const testText = testTexts[idx];
    const testVector = encodedTest[idx];
    const trueLabel = testLabels[idx];
    
    console.log(`Input: "${testText}" (true: ${trueLabel})`);
    
    try {
      const elmPredArray = elm.predictFromVector([testVector], uniqueLabels.length);
      const elmPred = elmPredArray[0] && elmPredArray[0][0] ? elmPredArray[0][0] : null;
      if (elmPred) {
        const elmConf = elmPred.prob ?? elmPred.confidence ?? 0;
        const elmPercent = (elmConf != null && !isNaN(elmConf) && isFinite(elmConf))
          ? `${(elmConf * 100).toFixed(2)}%`
          : 'N/A';
        console.log(`  ELM:        ${elmPred.label} (${elmPercent})`);
      } else {
        console.log(`  ELM:        Error - No prediction`);
      }
    } catch (error) {
      console.log(`  ELM:        Error - ${error.message}`);
    }
    
    try {
      const kelmProbs = kelm.predictProbaFromVectors([testVector])[0];
      const kelmIdx = kelmProbs.indexOf(Math.max(...kelmProbs));
      const kelmConf = kelmProbs[kelmIdx];
      console.log(`  KELM:       ${uniqueLabels[kelmIdx]} (${(kelmConf * 100).toFixed(2)}%)`);
    } catch (error) {
      console.log(`  KELM:       Error - ${error.message}`);
    }
    
    try {
      const ensemblePred = getEnsemblePredictionFromVector(ensemble, testVector, 1, 0.6)[0];
      const ensConf = ensemblePred.prob ?? 0;
      const ensPercent = (ensConf != null && !isNaN(ensConf) && isFinite(ensConf))
        ? `${(ensConf * 100).toFixed(2)}%`
        : 'N/A';
      console.log(`  Ensemble:   ${ensemblePred.label} (${ensPercent})`);
    } catch (error) {
      console.log(`  Ensemble:   Error - ${error.message}`);
    }
    console.log('');
  }

  return { elm, kelm, ensemble, encoder };
}

if (import.meta.url === `file://${process.argv[1]}` || 
    import.meta.url.endsWith('02-ensemble-classification.js') ||
    process.argv[1]?.endsWith('02-ensemble-classification.js')) {
  runEnsembleExample().catch(console.error);
}

export { 
  buildSharedEncoder,
  trainELMFromVectors,
  trainKernelELMFromVectors,
  getEnsemblePredictionFromVector,
  predictText,
  testELMAndCollectResultsVector,
  testKELMAndCollectResultsVector,
  testEnsembleAndCollectResultsVector,
  runEnsembleExample
};
