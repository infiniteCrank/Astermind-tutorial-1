// Test all examples
import { bootstrapELM } from './01-bootstrap.js';
import { runEnsembleExample } from './02-ensemble-classification.js';
import { createChainedRegression } from './03-chained-regression.js';

async function testAll() {
  console.log('🧪 Running All Tutorial Examples\n');
  console.log('=' .repeat(60) + '\n');

  try {
    console.log('1️⃣  Bootstrap Example\n');
    await bootstrapELM();
    console.log('\n' + '='.repeat(60) + '\n');

    console.log('2️⃣  Ensemble Classification Example\n');
    await runEnsembleExample();
    console.log('\n' + '='.repeat(60) + '\n');

    console.log('3️⃣  Chained Regression Example\n');
    await createChainedRegression();
    console.log('\n' + '='.repeat(60) + '\n');

    console.log('✅ All examples completed successfully!');
  } catch (error) {
    console.error('❌ Error running examples:', error);
    process.exit(1);
  }
}

if (import.meta.url === `file://${process.argv[1]}` || 
    import.meta.url.endsWith('04-test-all.js') ||
    process.argv[1]?.endsWith('04-test-all.js')) {
  testAll().catch(console.error);
}

export { testAll };

