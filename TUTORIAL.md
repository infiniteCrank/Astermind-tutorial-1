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
├── examples/
│   ├── 01-bootstrap.js
│   ├── 02-ensemble-classification.js
│   ├── 03-chained-regression.js
│   └── 04-test-all.js
└── README.md
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

After verifying your email, set the token:

```bash
export ASTERMIND_LICENSE_TOKEN="your-token-here"
```

### Basic Bootstrap Example

Let's create a bootstrap script that generates synthetic data and trains an ELM model:

```javascript
// examples/01-bootstrap.js
import { loadPretrained } from '@astermind/astermind-synth';
import { ELM } from '@astermind/astermind-elm';

async function bootstrapELM() {
  console.log('🚀 Bootstrapping ELM with AsterMind Synth...\n');

  // Load pretrained synth model
  const synth = loadPretrained('hybrid');
  
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
  const testCases = [
    await synth.generate('first_name'),
    await synth.generate('email'),
    await synth.generate('phone_number')
  ];

  for (const testCase of testCases) {
    const prediction = elm.predict(testCase, 3);
    console.log(`  Input: "${testCase}"`);
    console.log(`  Prediction: ${prediction.map(p => `${p.label} (${(p.confidence * 100).toFixed(2)}%)`).join(', ')}\n`);
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

Let's create an ensemble that combines ELM and KELM models using proper probability fusion:

```javascript
// examples/02-ensemble-classification.js
import { ELM, KernelELM } from '@astermind/astermind-elm';
import { loadPretrained } from '@astermind/astermind-synth';

/**
 * Ensemble model structure matching the official implementation.
 * Uses full probability fusion for better accuracy.
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
   * 
   * Key points:
   * - Gets probabilities for ALL labels from both models (not just topK)
   * - Uses weighted fusion: elmWeight * elmProbs + kelmWeight * kelmProbs
   * - Normalizes after fusion to ensure probabilities sum to 1
   * - Then takes topK after fusion
   */
  predictFromVector(x, topK = 3, kelmWeight = 0.6) {
    const elmWeight = 1 - kelmWeight;

    // ELM probabilities - get ALL labels
    const elmProbsArr = this.elm.predictFromVector([x], this.uniqueLabels.length)[0];
    const elmProbs = new Array(this.uniqueLabels.length).fill(0);
    for (const p of elmProbsArr) {
      const idx = this.uniqueLabels.indexOf(p.label);
      if (idx >= 0) elmProbs[idx] = p.prob || p.confidence || 0;
    }

    // KernelELM probabilities
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

  // Convenience method for text input
  predict(text, topK = 3, kelmWeight = 0.6) {
    const encoded = this.encoder.encode(text);
    const normalized = this.encoder.normalize(encoded);
    return this.predictFromVector(normalized, topK, kelmWeight);
  }
}

async function runEnsembleExample() {
  console.log('🎯 Ensemble Classification Example\n');

  // Generate training data
  const synth = loadPretrained('hybrid');
  await new Promise(resolve => setTimeout(resolve, 100));

  const categories = ['first_name', 'last_name', 'email', 'phone_number'];
  const texts = [];
  const labels = [];

  console.log('📊 Generating training data...');
  for (let i = 0; i < 200; i++) {
    for (const category of categories) {
      const value = await synth.generate(category);
      texts.push(value);
      labels.push(category);
    }
  }

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

  // Train KELM (using vectorized inputs)
  console.log('🎓 Training KELM...');
  const kelm = new KernelELM({
    outputDim: categories.length,
    kernel: { type: 'rbf', gamma: 1.0 / encodedTexts[0].length },
    mode: 'nystrom',
    nystrom: { m: 128, strategy: 'random', whiten: false },
    ridgeLambda: 1e-2
  });

  // Convert labels to one-hot
  const oneHotLabels = labelIndices.map(idx => {
    const oneHot = new Array(categories.length).fill(0);
    oneHot[idx] = 1;
    return oneHot;
  });

  kelm.fit(encodedTexts, oneHotLabels);

  // Create ensemble
  console.log('🔗 Creating ensemble...');
  const ensemble = new EnsembleClassifier([elm, kelm]);

  // Test individual models vs ensemble
  console.log('\n🧪 Testing Models:\n');
  const testCases = [
    await synth.generate('first_name'),
    await synth.generate('email'),
    await synth.generate('phone_number')
  ];

  for (const testCase of testCases) {
    console.log(`Input: "${testCase}"`);
    
    const elmPred = elm.predict(testCase, 1)[0];
    console.log(`  ELM:        ${elmPred.label} (${(elmPred.confidence * 100).toFixed(2)}%)`);
    
    // For KELM, we need to encode and predict
    const encoded = elm.encoder.encode(testCase);
    const normalized = elm.encoder.normalize(encoded);
    const kelmProbs = kelm.predictProbaFromVectors([normalized])[0];
    const kelmIdx = kelmProbs.indexOf(Math.max(...kelmProbs));
    const kelmConf = kelmProbs[kelmIdx];
    console.log(`  KELM:       ${categories[kelmIdx]} (${(kelmConf * 100).toFixed(2)}%)`);
    
    const ensemblePred = ensemble.predict(testCase, 1)[0];
    console.log(`  Ensemble:   ${ensemblePred.label} (${(ensemblePred.confidence * 100).toFixed(2)}%)`);
    console.log('');
  }

  return { elm, kelm, ensemble };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  runEnsembleExample().catch(console.error);
}

export { EnsembleClassifier, runEnsembleExample };
```

Run this example:

```bash
npm run ensemble
```

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

/**
 * Chained Regression Example
 * 
 * Problem: Predict user engagement score from text features
 * 
 * Stage 1: ELM1 predicts intermediate features (sentiment, length, complexity)
 * Stage 2: ELM2 uses these features to predict final engagement score
 */

async function createChainedRegression() {
  console.log('🔗 Chained Regression Example\n');

  // Generate synthetic training data
  const synth = loadPretrained('hybrid');
  await new Promise(resolve => setTimeout(resolve, 100));

  // Generate diverse text samples
  const texts = [];
  for (let i = 0; i < 500; i++) {
    const category = ['first_name', 'last_name', 'email', 'company_name'][
      Math.floor(Math.random() * 4)
    ];
    const text = await synth.generate(category);
    texts.push(text);
  }

  // Create synthetic intermediate features (what ELM1 should learn to predict)
  // These represent: [sentiment_score, length_score, complexity_score]
  const intermediateFeatures = texts.map(text => {
    const sentiment = Math.sin(text.length * 0.1) * 0.5 + 0.5; // 0-1
    const length = Math.min(text.length / 50, 1.0); // normalized length
    const complexity = (text.match(/[A-Z]/g) || []).length / text.length; // capital ratio
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

  console.log('📊 Generated training data:');
  console.log(`  Texts: ${texts.length}`);
  console.log(`  Intermediate features: ${intermediateFeatures[0].length} per sample`);
  console.log(`  Target scores: ${engagementScores.length}\n`);

  // Stage 1: Train ELM1 to predict intermediate features from text
  console.log('🎓 Stage 1: Training ELM1 (Text → Intermediate Features)...');
  const elm1 = new ELM({
    categories: ['feature1', 'feature2', 'feature3'], // dummy categories for regression
    hiddenUnits: 256,
    maxLen: 50,
    useTokenizer: true,
    activation: 'relu',
    ridgeLambda: 1e-4
  });

  // Encode texts
  const encodedTexts = texts.map(text => {
    const encoded = elm1.encoder.encode(text);
    return elm1.encoder.normalize(encoded);
  });

  // Train ELM1 for regression (treating each feature as a separate output)
  // We'll train 3 separate ELMs for each feature, or use a multi-output approach
  const elm1Features = [];
  for (let featIdx = 0; featIdx < 3; featIdx++) {
    const featureELM = new ELM({
      categories: ['output'],
      hiddenUnits: 128,
      maxLen: 50,
      useTokenizer: true,
      activation: 'relu',
      ridgeLambda: 1e-4
    });

    const targetValues = intermediateFeatures.map(f => [f[featIdx]]);
    featureELM.trainFromData(encodedTexts, targetValues.map(v => [v[0]]));
    elm1Features.push(featureELM);
  }

  console.log('✅ ELM1 training complete\n');

  // Generate intermediate predictions
  const predictedIntermediate = encodedTexts.map(encoded => {
    const features = elm1Features.map(elm => {
      const pred = elm.predictFromVector(encoded, 1);
      return pred[0]?.confidence || 0;
    });
    return features;
  });

  console.log('📈 Sample intermediate predictions:');
  console.log(`  Actual:    [${intermediateFeatures[0].map(f => f.toFixed(3)).join(', ')}]`);
  console.log(`  Predicted: [${predictedIntermediate[0].map(f => f.toFixed(3)).join(', ')}]\n`);

  // Stage 2: Train ELM2 to predict engagement from intermediate features
  console.log('🎓 Stage 2: Training ELM2 (Intermediate Features → Engagement Score)...');
  
  // Create a simple ELM for regression
  const elm2 = new ELM({
    categories: ['engagement'],
    hiddenUnits: 64,
    maxLen: 3, // input is 3 features
    useTokenizer: false, // numeric input, no tokenization
    activation: 'relu',
    ridgeLambda: 1e-4
  });

  // Convert intermediate features to text-like format for ELM
  // (ELM expects text input, so we'll encode the features as strings)
  const featureTexts = predictedIntermediate.map(feat => 
    feat.map(f => f.toFixed(6)).join(' ')
  );

  const encodedFeatures = featureTexts.map(text => {
    const encoded = elm2.encoder.encode(text);
    return elm2.encoder.normalize(encoded);
  });

  const targetScores = engagementScores.map(s => [s[0]]);
  elm2.trainFromData(encodedFeatures, targetScores);

  console.log('✅ ELM2 training complete\n');

  // Test the chained model
  console.log('🧪 Testing Chained Model:\n');
  
  const testTexts = texts.slice(0, 5);
  const testActualIntermediate = intermediateFeatures.slice(0, 5);
  const testActualScores = engagementScores.slice(0, 5);

  for (let i = 0; i < testTexts.length; i++) {
    const text = testTexts[i];
    const actualIntermediate = testActualIntermediate[i];
    const actualScore = testActualScores[i][0];

    // Stage 1: Text → Intermediate Features
    const encoded = elm1.encoder.encode(text);
    const normalized = elm1.encoder.normalize(encoded);
    const predictedFeat = elm1Features.map(elm => {
      const pred = elm.predictFromVector(normalized, 1);
      return pred[0]?.confidence || 0;
    });

    // Stage 2: Intermediate Features → Engagement Score
    const featText = predictedFeat.map(f => f.toFixed(6)).join(' ');
    const encodedFeat = elm2.encoder.encode(featText);
    const normalizedFeat = elm2.encoder.normalize(encodedFeat);
    const predictedScore = elm2.predictFromVector(normalizedFeat, 1);
    const finalScore = predictedScore[0]?.confidence || 0;

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
    const predictedFeat = elm1Features.map(elm => {
      const pred = elm.predictFromVector(normalized, 1);
      return pred[0]?.confidence || 0;
    });
    const featText = predictedFeat.map(f => f.toFixed(6)).join(' ');
    const encodedFeat = elm2.encoder.encode(featText);
    const normalizedFeat = elm2.encoder.normalize(encodedFeat);
    const predictedScore = elm2.predictFromVector(normalizedFeat, 1);
    const finalScore = predictedScore[0]?.confidence || 0;
    totalError += Math.abs(engagementScores[i][0] - finalScore);
  }

  const mae = totalError / texts.length;
  console.log(`📊 Overall Mean Absolute Error: ${mae.toFixed(4)}`);

  return { elm1Features, elm2 };
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

