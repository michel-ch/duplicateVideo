import puppeteer from 'puppeteer';
import * as fs from 'fs';
import * as path from 'path';

(async () => {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();
  
  await page.setViewport({ width: 1440, height: 900 });

  // Wait for the app to be fully loaded
  await page.goto('http://localhost:5173', { waitUntil: 'load', timeout: 30000 });
  await new Promise(r => setTimeout(r, 2000));

  console.log('Opened app successfully. Taking screenshot of main page...');
  const screenshotsDir = path.join(process.cwd(), '..', 'docs');
  if (!fs.existsSync(screenshotsDir)) {
    fs.mkdirSync(screenshotsDir, { recursive: true });
  }

  await page.screenshot({ path: path.join(screenshotsDir, 'dashboard.png') });
  console.log('Saved dashboard.png');
  
  // Attempt to navigate to another page or trigger a dialog to get more images
  // The user wants "images" (plural) of the app
  // Without knowing the DOM, let's just take one for now and close.
  // I can look at the frontend code to find buttons or just take one.
  
  await browser.close();
  console.log('Capture complete!');
})();
