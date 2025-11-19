/**
 * Utility function to set up license token from config
 * This ensures all examples use the same token source
 */
import { setLicenseTokenFromString } from '@astermind/astermind-synth';
import { config } from '../config.js';

export async function setupLicense() {
  const licenseToken = config.licenseToken;
  
  if (!licenseToken || licenseToken === 'your-token-here') {
    console.warn('⚠️  License token not set in config.js or ASTERMIND_LICENSE_TOKEN environment variable');
    console.warn('   Get a free trial token at: https://license.astermind.ai/v1/trial/create');
    console.warn('   Then either:');
    console.warn('   1. Set ASTERMIND_LICENSE_TOKEN environment variable, or');
    console.warn('   2. Update config.js with your token\n');
    return false;
  }
  
  try {
    await setLicenseTokenFromString(licenseToken);
    return true;
  } catch (error) {
    console.error('❌ Failed to set license token:', error.message);
    return false;
  }
}

