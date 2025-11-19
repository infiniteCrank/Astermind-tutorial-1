/**
 * Configuration template for AsterMind Tutorial
 * 
 * Copy this file to config.js and update with your license token:
 *   cp config.example.js config.js
 * 
 * Then edit config.js and replace 'your-token-here' with your actual token.
 */

export const config = {
  // Put your license token here
  // You can get a free 30-day trial token at: https://license.astermind.ai/v1/trial/create
  licenseToken: process.env.ASTERMIND_LICENSE_TOKEN || 'your-token-here',
  
  // Other configuration options can go here
  synthMode: 'hybrid', // 'retrieval', 'elm', 'hybrid', 'exact', 'premium'
};

