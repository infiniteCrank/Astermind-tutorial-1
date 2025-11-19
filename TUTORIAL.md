# AsterMind ELM Tutorial: Ensemble Methods and Chained Regression

Welcome to this comprehensive tutorial on using **AsterMind ELM** and **AsterMind Synth** to build powerful machine learning models. This tutorial covers:

1. **Bootstrapping your project** with AsterMind Synth
2. **Ensemble methods** combining ELM and KELM models
3. **Why ensemble methods outperform standalone models** for classification
4. **Chained regression** - connecting ELMs to solve complex problems

---

## Table of Contents

1. [Introduction](#introduction)
2. [Setting Up Your Project](#setting-up-your-project)
3. [Bootstrapping with AsterMind Synth](#bootstrapping-with-astermind-synth)
4. [Understanding ELM and KELM](#understanding-elm-and-kelm)
5. [Ensemble Methods for Classification](#ensemble-methods-for-classification)
6. [Why Ensemble Methods Work Better](#why-ensemble-methods-work-better)
7. [Chained Regression: ELMs Working Together](#chained-regression-elms-working-together)
8. [Conclusion](#conclusion)

---

## Introduction

**Extreme Learning Machines (ELMs)** are a class of feedforward neural networks that train extremely fast by using random hidden layer weights and solving only the output layer analytically. **Kernel ELMs (KELMs)** extend ELMs with kernel functions to handle non-linear data more effectively.

**AsterMind Synth** is a synthetic data generator that can bootstrap your ELM projects by generating realistic training data, while **AsterMind ELM** provides the core machine learning capabilities.

### Key Concepts

- **ELM**: Fast training, good for linear and simple non-linear patterns
- **KELM**: Better for complex non-linear patterns using kernel functions
- **Ensemble Methods**: Combine multiple models to improve accuracy and robustness
- **Chained ELMs**: Use one ELM's output as another's input for hierarchical learning

---

## Setting Up Your Project

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn package manager

### Installation

First, create a new directory and initialize your project:

```bash
mkdir astermind-elm-tutorial
cd astermind-elm-tutorial
npm init -y
```

Install the required packages:

```bash
npm install @astermind/astermind-elm @astermind/astermind-synth
```

### Project Structure

```
astermind-elm-tutorial/
├── package.json
├── TUTORIAL.md
├── README.md
├── config.example.js          # Template for config.js
├── config.js                  # Your license token (gitignored)
├── utils/
│   └── setupLicense.js       # License setup utility
└── examples/
    ├── 00-check-setup.js
    ├── 01-bootstrap.js
    ├── 02-ensemble-classification.js
    ├── 03-chained-regression.js
    └── 04-test-all.js
```

---

## Bootstrapping with AsterMind Synth

AsterMind Synth allows you to quickly generate synthetic training data for your ELM models. This is especially useful when you need to bootstrap a project or augment existing datasets.

### Getting a License Token

AsterMind Synth requires a license token. You can get a free 30-day trial:

```bash
curl -X POST "https://license.astermind.ai/v1/trial/create" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your-email@example.com",
    "product": "astermind-synth"
  }'
```

After verifying your email, you can set the token in two ways:

**Option 1: Use config.js (Recommended)**

Copy the example config file:
```bash
cp config.example.js config.js
```

Then edit `config.js` and add your token:
```javascript
export const config = {
  licenseToken: 'your-token-here',
  synthMode: 'hybrid', // 'retrieval', 'elm', 'hybrid', 'exact', 'premium'
};
```

**Option 2: Use environment variable**

```bash
export ASTERMIND_LICENSE_TOKEN="your-token-here"
```

The environment variable takes precedence if both are set. All examples automatically read from `config.js` or the environment variable.

### Basic Bootstrap Example

Let's create a bootstrap script that generates synthetic data and trains an ELM model:

```javascript
// examples/01-bootstrap.js
import { loadPretrained } from '@astermind/astermind-synth';
import { ELM } from '@astermind/astermind-elm';
import { setupLicense } from '../utils/setupLicense.js';
import { config } from '../config.js';

async function bootstrapELM() {
  // Set up license token from config (must be done before using synth)
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
      const value = await synth.generate(category);
      texts.push(value);
      labels.push(category);
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
        let conf = p.confidence ?? p.prob;
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
if (import.meta.url === `file://${process.argv[1]}`) {
  bootstrapELM().catch(console.error);
}

export { bootstrapELM };
```

Run this example:

```bash
npm run bootstrap
```

---

## Understanding ELM and KELM

### ELM (Extreme Learning Machine)

ELMs use a single hidden layer with random weights. Only the output layer is trained, making them extremely fast:

```javascript
import { ELM } from '@astermind/astermind-elm';

const elm = new ELM({
  categories: ['A', 'B', 'C'],
  hiddenUnits: 128,
  activation: 'relu',
  ridgeLambda: 1e-4
});
```

### KELM (Kernel ELM)

KELMs use kernel functions to map data to a higher-dimensional space, enabling better handling of non-linear patterns:

```javascript
import { KernelELM, KernelRegistry } from '@astermind/astermind-elm';

const kelm = new KernelELM({
  outputDim: 3, // number of classes
  kernel: { type: 'rbf', gamma: 0.1 },
  mode: 'exact',
  ridgeLambda: 1e-2
});
```

---

## Ensemble Methods for Classification

Ensemble methods combine multiple models to improve predictive performance. They often outperform standalone models by:

1. **Reducing variance**: Averaging predictions from multiple models reduces overfitting
2. **Reducing bias**: Different models capture different aspects of the data
3. **Improving robustness**: Errors in one model are compensated by others

### Creating an Ensemble

Let's create an ensemble that combines ELM and KELM models using proper probability fusion. This example follows a vector-based pipeline with shared encoding and train/test evaluation:

```javascript
// examples/02-ensemble-classification.js
import { ELM, KernelELM } from '@astermind/astermind-elm';
import { loadPretrained } from '@astermind/astermind-synth';
import { setupLicense } from '../utils/setupLicense.js';
import { config } from '../config.js';

/**
 * Build a shared encoder for text-to-vector conversion
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
  
  if (elm.setCategories) {
    elm.setCategories(uniqueLabels);
  }
  
  return elm.encoder;
}

/**
 * Train ELM from pre-encoded vectors
 */
function trainELMFromVectors(X, labels, uniqueLabels, config = {}) {
  const {
    hiddenUnits = 128,
    activation = 'relu',
    ridgeLambda = 1e-4
  } = config;
  
  const elm = new ELM({
    categories: uniqueLabels,
    hiddenUnits: hiddenUnits,
    maxLen: X[0].length,
    useTokenizer: false,
    activation: activation,
    ridgeLambda: ridgeLambda
  });
  
  if (elm.setCategories) {
    elm.setCategories(uniqueLabels);
  }
  
  const labelIndices = labels.map(l => uniqueLabels.indexOf(l));
  elm.trainFromData(X, labelIndices);
  
  return elm;
}

/**
 * Train KernelELM from pre-encoded vectors with data-driven gamma
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
  let gamma = 1.0 / X[0].length;
  if (kernelType === 'rbf' && X.length > 1) {
    const distances = [];
    const sampleSize = Math.min(100, X.length);
    
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
 * Uses full probability fusion: elmWeight * pELM + kelmWeight * pKELM
 */
function getEnsemblePredictionFromVector(ensemble, x, topK = 3, kelmWeight = 0.6) {
  const elmWeight = 1 - kelmWeight;
  
  // Get ELM probabilities for all labels
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
 * Ensemble model structure: { elm, kelm, encoder, uniqueLabels }
 */
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
        // Fallback synthetic data
        if (category === 'first_name') texts.push(`Name${i}`);
        else if (category === 'last_name') texts.push(`Surname${i}`);
        else if (category === 'email') texts.push(`user${i}@example.com`);
        else if (category === 'phone_number') texts.push(`555-${1000 + i}`);
        labels.push(category);
      }
    }
  }

  console.log(`✅ Generated ${texts.length} samples\n`);

  // Split into train/test sets (80/20 per label to maintain balance)
  console.log('📊 Splitting data into train/test sets...');
  const uniqueLabelsSet = [...new Set(labels)];
  const trainTexts = [];
  const trainLabels = [];
  const testTexts = [];
  const testLabels = [];
  
  for (const label of uniqueLabelsSet) {
    const indices = labels.map((l, idx) => l === label ? idx : -1).filter(idx => idx >= 0);
    const testCount = Math.floor(indices.length * 0.2);
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
  
  // Simplified test helpers (compute accuracy)
  let elmCorrect = 0, kelmCorrect = 0, ensembleCorrect = 0;
  for (let i = 0; i < encodedTest.length; i++) {
    const x = encodedTest[i];
    const trueLabel = testLabels[i];
    
    // ELM prediction
    const elmPredArray = elm.predictFromVector([x], uniqueLabels.length);
    const elmPred = elmPredArray[0] && elmPredArray[0][0] ? elmPredArray[0][0] : null;
    if (elmPred?.label === trueLabel) elmCorrect++;
    
    // KELM prediction
    const kelmProbs = kelm.predictProbaFromVectors([x])[0];
    const kelmIdx = kelmProbs.indexOf(Math.max(...kelmProbs));
    if (uniqueLabels[kelmIdx] === trueLabel) kelmCorrect++;
    
    // Ensemble prediction
    const ensemblePred = getEnsemblePredictionFromVector(ensemble, x, 1, 0.6)[0];
    if (ensemblePred?.label === trueLabel) ensembleCorrect++;
  }
  
  const elmAccuracy = elmCorrect / encodedTest.length;
  const kelmAccuracy = kelmCorrect / encodedTest.length;
  const ensembleAccuracy = ensembleCorrect / encodedTest.length;

  // Print comparison report
  console.log('========================');
  console.log('Model Comparison (Form Fields)');
  console.log('========================');
  console.log(`ELM:        ${elmCorrect}/${encodedTest.length} (${(elmAccuracy * 100).toFixed(2)}%)`);
  console.log(`KernelELM:  ${kelmCorrect}/${encodedTest.length} (${(kelmAccuracy * 100).toFixed(2)}%)`);
  console.log(`Ensemble:   ${ensembleCorrect}/${encodedTest.length} (${(ensembleAccuracy * 100).toFixed(2)}%)`);
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

if (import.meta.url === `file://${process.argv[1]}`) {
  runEnsembleExample().catch(console.error);
}

export { 
  buildSharedEncoder,
  trainELMFromVectors,
  trainKernelELMFromVectors,
  getEnsemblePredictionFromVector,
  runEnsembleExample
};
```

Run this example:

```bash
npm run ensemble
```

**Expected Output:**

The example will:
1. Generate 800 synthetic samples (200 per category)
2. Split into train/test sets (80/20 per label)
3. Build a shared encoder for text-to-vector conversion
4. Train ELM and KernelELM from pre-encoded vectors
5. Create an ensemble using probability fusion
6. Evaluate all models on the held-out test set
7. Display a comparison report showing accuracy for each model

**Example Output:**
```
========================
Model Comparison (Form Fields)
========================
ELM:        137/160 (85.63%)
KernelELM:  137/160 (85.63%)
Ensemble:   141/160 (88.13%)
========================
```

The ensemble typically outperforms individual models by 2-5% accuracy, demonstrating the power of combining multiple models.

---

## Why Ensemble Methods Work Better

Ensemble methods often outperform standalone ELM and KELM models for several reasons:

### 1. **Bias-Variance Tradeoff**

- **ELM models** can have high variance due to random initialization
- **KELM models** can have high bias if the kernel parameters aren't optimal
- **Ensembles** balance both by averaging predictions

### 2. **Diversity in Learning**

Different models capture different patterns:
- **ELM**: Good at capturing linear and simple non-linear relationships
- **KELM**: Better at complex non-linear patterns through kernel mapping
- **Combined**: Covers a wider range of patterns

### 3. **Error Reduction**

If one model makes an error, others can compensate:
- Individual model accuracy: ~75-85%
- Ensemble accuracy: Often 85-95%+

### 4. **Robustness to Overfitting**

Ensembles are less prone to overfitting because:
- Multiple models reduce the impact of any single model's overfitting
- Averaging smooths out extreme predictions

### 5. **Statistical Theory**

The **Central Limit Theorem** suggests that averaging multiple independent estimates reduces variance, leading to more stable predictions.

---

## Chained Regression: ELMs Working Together

Chaining ELMs allows you to solve complex regression tasks by breaking them into hierarchical steps. The output of one ELM becomes the input to another, enabling the modeling of intricate relationships.

### Use Case: Multi-Stage Prediction

Consider a scenario where you need to predict a complex value that depends on intermediate representations:

```javascript
// examples/03-chained-regression.js
import { ELM } from '@astermind/astermind-elm';
import { loadPretrained } from '@astermind/astermind-synth';
import { setupLicense } from '../utils/setupLicense.js';
import { config } from '../config.js';

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
  // Set up license token from config (must be done before using synth)
  await setupLicense();
  
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

if (import.meta.url === `file://${process.argv[1]}`) {
  createChainedRegression().catch(console.error);
}

export { createChainedRegression };
```

Run this example:

```bash
npm run chain
```

**Expected Output:**
```
🔗 Chained Regression Example

📊 Generating training data...
✅ Generated 500 training samples
  Intermediate features: 3 per sample
  Target scores: 500

🎓 Stage 1a: Building text encoder...
✅ Encoder ready (vector dimension: 50)

🎓 Stage 1b: Training feature ELMs (Encoded Vector → Binned Features)...
✅ Feature ELMs training complete

📈 Sample intermediate predictions:
  Actual:    [0.996, 0.340, 0.353]
  Predicted: [0.965, 0.312, 0.400]

🎓 Stage 2: Training ELM2 (Intermediate Features → Engagement Score)...
✅ ELM2 training complete

🧪 Testing Chained Model:

Text: "john.doe@email.com"
  Intermediate Features:
    Actual:    [0.987, 0.360, 0.000]
    Predicted: [0.965, 0.348, 0.000]
  Engagement Score:
    Actual:    0.5629
    Predicted: 0.5657
    Error:     0.0028

📊 Overall Mean Absolute Error: 0.0484
```

### Key Technical Details

This example demonstrates several important techniques:

1. **Regression via Classification Bins**: Since ELMs are classification models, we approximate regression by:
   - Binning continuous values into discrete classes (e.g., 10 bins)
   - Training classification ELMs to predict the bin
   - Mapping predicted bins back to continuous values
   - This keeps the API purely in classification mode while behaving like a coarse regressor

2. **Vector-Based Training**: Both feature ELMs and ELM2 use `useTokenizer: false` with `inputSize` (not `maxLen`) to work directly with numerical vectors rather than text, enabling efficient chaining.

3. **Robust Encoder Access**: The encoder is extracted from `encoderELM` with a guard check and reused throughout, avoiding undefined access issues.

4. **Numerically Safe Binning**: Binning logic includes guards against division by zero for degenerate cases where all values are equal.

5. **Prediction Access Pattern**: The `predictFromVector` method expects an array of vectors `[vector]` and returns `[[{label, prob}, ...]]`, so we access predictions as `predArray[0][0]`.

6. **Direct Feature Passing**: ELM2 receives the intermediate features directly as a 3D vector `[sentiment, length, complexity]`, avoiding unnecessary text encoding.

7. **Chained Training**: Stage 2 trains on predicted intermediate features (not ground truth), ensuring the model reflects real chained performance rather than an oracle scenario.

### Why Chaining Works

1. **Hierarchical Learning**: Each ELM focuses on a specific level of abstraction
2. **Feature Extraction**: Early ELMs act as feature extractors
3. **Complex Relationships**: Later ELMs can model complex interactions between extracted features
4. **Modularity**: Each stage can be optimized independently

---

## Conclusion

In this tutorial, we've covered:

1. ✅ **Bootstrapping projects** with AsterMind Synth
2. ✅ **Creating ensemble models** combining ELM and KELM
3. ✅ **Understanding why ensembles work better** than standalone models
4. ✅ **Chaining ELMs** for complex regression tasks

### Key Takeaways

- **Ensemble methods** reduce variance and bias, leading to better generalization
- **ELM and KELM** complement each other in ensembles
- **Chained ELMs** enable hierarchical learning for complex problems
- **AsterMind Synth** provides quick data generation for bootstrapping

### Next Steps

- Experiment with different ensemble weights
- Try different kernel functions in KELM
- Explore deeper chains (3+ ELMs)
- Fine-tune hyperparameters (hidden units, ridge lambda, etc.)

### Resources

- [AsterMind ELM Documentation](https://www.npmjs.com/package/@astermind/astermind-elm)
- [AsterMind Synth Documentation](https://www.npmjs.com/package/@astermind/astermind-synth)
- [GitHub Repository](https://github.com/infiniteCrank/AsterMind-ELM)

---

**Happy Learning! 🚀**

