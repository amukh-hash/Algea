const puppeteer = require('puppeteer');
const path = require('path');

(async () => {
    // 1. Setup paths
    // Use the absolute path to the HTML file we created earlier
    const htmlPath = 'C:/Users/crick/.gemini/antigravity/brain/65e2f529-31c4-4c3b-ade1-08e50661883c/architecture.html';
    const pdfPath = 'C:/Users/crick/.gemini/antigravity/brain/65e2f529-31c4-4c3b-ade1-08e50661883c/architecture_rendered.pdf';

    console.log(`Loading ${htmlPath}...`);

    try {
        const browser = await puppeteer.launch();
        const page = await browser.newPage();

        // 2. Load the page
        // Ensure we load as a file:// URL
        await page.goto(`file://${htmlPath}`, { waitUntil: 'networkidle0' });

        // 3. Wait for Mermaid to render
        // We look for the SVG elements that mermaid generates
        try {
            await page.waitForSelector('svg', { timeout: 5000 });
            console.log('Mermaid diagrams detected.');
        } catch (e) {
            console.warn('Timeout waiting for SVG selector. Proceeding anyway, diagrams might not be fully rendered if network is slow.');
        }

        // 4. Generate PDF
        console.log(`Printing to ${pdfPath}...`);
        await page.pdf({
            path: pdfPath,
            format: 'A4',
            printBackground: true,
            margin: { top: '20px', right: '20px', bottom: '20px', left: '20px' }
        });

        await browser.close();
        console.log('Done.');

    } catch (err) {
        console.error('Error generating PDF:', err);
        process.exit(1);
    }
})();
