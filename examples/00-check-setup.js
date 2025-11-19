// Setup check script - verifies dependencies and license token
import { config } from '../config.js';

console.log('🔍 Checking setup...\n');

// Check if packages are installed
try {
  const elm = await import('@astermind/astermind-elm');
  console.log('✅ @astermind/astermind-elm installed');
  console.log(`   Available exports: ${Object.keys(elm).join(', ')}\n`);
} catch (error) {
  console.error('❌ @astermind/astermind-elm not found:', error.message);
  process.exit(1);
}

// Check synth package
try {
  const synth = await import('@astermind/astermind-synth');
  console.log('✅ @astermind/astermind-synth installed');
  console.log(`   Available exports: ${Object.keys(synth).join(', ')}\n`);
} catch (error) {
  console.warn('⚠️  @astermind/astermind-synth import issue:', error.message);
  console.warn('   The examples will use fallback data if synth fails.\n');
}

// Check license token (from config or environment)
const licenseToken = config.licenseToken;
if (licenseToken && licenseToken !== 'your-token-here') {
  console.log('✅ License token found');
  if (process.env.ASTERMIND_LICENSE_TOKEN) {
    console.log('   Source: Environment variable (ASTERMIND_LICENSE_TOKEN)');
  } else {
    console.log('   Source: config.js');
  }
  console.log(`   Token: ${licenseToken.substring(0, 20)}...\n`);
} else {
  console.warn('⚠️  License token not set');
  console.warn('   Options:');
  console.warn('   1. Set ASTERMIND_LICENSE_TOKEN environment variable, or');
  console.warn('   2. Update config.js with your token');
  console.warn('   Get a free trial token at: https://license.astermind.ai');
  console.warn('   Or use: julian@astermind.ai for testing\n');
}

// Check config
console.log('📋 Configuration:');
console.log(`   Synth mode: ${config.synthMode}`);
console.log('');

console.log('✅ Setup check complete!\n');

