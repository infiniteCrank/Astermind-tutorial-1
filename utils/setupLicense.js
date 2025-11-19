/**
 * Utility function to set up license token from config
 * This ensures all examples use the same token source
 * 
 * IMPORTANT: This must be called BEFORE importing @astermind/astermind-synth
 * because the library checks the environment variable at module initialization
 */
export async function setupLicense() {
  // Dynamically import config to avoid top-level import issues
  const { config } = await import('../config.js');
  const licenseToken = config.licenseToken;
  
  if (!licenseToken || licenseToken === 'your-token-here') {
    console.warn('⚠️  License token not set in config.js or ASTERMIND_LICENSE_TOKEN environment variable');
    console.warn('   Get a free trial token at: https://license.astermind.ai/v1/trial/create');
    console.warn('   Then either:');
    console.warn('   1. Set ASTERMIND_LICENSE_TOKEN environment variable, or');
    console.warn('   2. Update config.js with your token\n');
    return false;
  }
  
  // Set environment variable (library reads this during initialization)
  // This MUST be set before importing @astermind/astermind-synth
  if (!process.env.ASTERMIND_LICENSE_TOKEN) {
    process.env.ASTERMIND_LICENSE_TOKEN = licenseToken;
  }
  
  // Note: The synth library will initialize the license runtime with "astermind-synth" audience
  // If your token has "astermind-elm" audience (but includes "astermind-synth" feature),
  // you may need a token with "astermind-synth" audience, or the library needs to be updated
  // to accept tokens with either audience as long as they have the required feature.
  
  return true;
}

