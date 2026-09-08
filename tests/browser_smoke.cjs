/* Start Streamlit on localhost:8517, then run with Playwright installed:
 * NODE_PATH=/path/to/node_modules node tests/browser_smoke.cjs
 * Optional: KG_BASE_URL, KG_CHROME, KG_SCREENSHOT.
 */
const { chromium } = require(process.argv[2] || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');

(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.KG_CHROME || '/usr/bin/google-chrome',
    headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage'],
  });
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', err => errors.push(err.message));
  const base = process.env.KG_BASE_URL || 'http://127.0.0.1:8517';
  // Confirm the component works without fetching any external scripts/assets.
  await context.route('**/*', route => {
    const url = route.request().url();
    if (url.startsWith(base) || url.startsWith('data:') || url.startsWith('blob:')) route.continue();
    else route.abort();
  });
  try {
    await page.goto(base);
    await page.getByRole('tab', { name: 'Knowledge Graph', exact: true }).click();
    const canvas = page.locator('.kg-canvas');
    await page.waitForFunction(() => {
      // The locator below is used for actual assertions; this only gives the
      // websocket its first render without relying on a fixed sleep.
      return document.body.textContent.includes('visible nodes');
    });
    await canvas.waitFor({ state: 'visible', timeout: 30000 });
    await page.locator('.kg-picker option').nth(1).waitFor({ state: 'attached' });
    assert.equal(errors.length, 0, errors.join('\n'));
    assert.ok(await canvas.locator('canvas').count() > 0, 'Cytoscape must render a canvas');
    assert.equal(await canvas.evaluate(el => el._cyreg.cy.nodes('[kind = "Measurement"]').length), 0,
      'Measurements must be absent initially, not merely dimmed');
    await page.getByRole('button', { name: 'Fit graph', exact: true }).click();
    await canvas.scrollIntoViewIfNeeded();

    const concept = await canvas.evaluate(el => {
      const cy = el._cyreg.cy;
      const node = cy.nodes().filter(n => n.data('kind') === 'Concept')[0];
      return { id: node.id(), position: node.renderedPosition(), label: node.data('label') };
    });
    const box = await canvas.boundingBox();
    await canvas.evaluate(el => {
      el._cyreg.cy.on('tap', event => { el.dataset.testTap = event.target.id?.() || 'background'; });
    });
    await page.mouse.click(box.x + concept.position.x, box.y + concept.position.y);
    assert.equal(await canvas.getAttribute('data-test-tap'), concept.id, 'Canvas click must reach the selected node');
    await page.getByRole('heading', { name: 'Selected evidence', exact: true }).waitFor();
    assert.ok(await page.locator('.kg-status').textContent());
    await page.getByRole('tab', { name: 'Annotated text', exact: true }).click();
    await page.waitForFunction(() => [...document.querySelectorAll('.ner-text mark')]
      .some(mark => mark.style.outlineWidth === '2px'));
    await page.getByRole('tab', { name: 'Tables', exact: true }).click();
    await page.getByText('Only graph selection', { exact: true }).click();
    assert.equal(await page.getByRole('checkbox', { name: 'Only graph selection', exact: true }).isChecked(), true);
    await page.getByRole('tab', { name: 'Knowledge Graph', exact: true }).click();

    const downloadPromise = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Download graph JSON', exact: true }).click();
    const download = await downloadPromise;
    const graph = JSON.parse(await fs.readFile(await download.path(), 'utf8'));
    const ids = new Set(graph.nodes.map(n => n.id));
    assert.ok(graph.edges.every(e => ids.has(e.source) && ids.has(e.target)));
    assert.equal(graph.view.max_nodes, 100);
    assert.ok(graph.nodes.some(n => n.id === concept.id));
    assert.equal(graph.schema_version, 2);
    assert.equal(graph.nodes.filter(n => n.kind === 'Person').length, 1);
    assert.ok(graph.nodes.some(n => n.kind === 'Age'));
    assert.ok(graph.nodes.some(n => n.kind === 'Sex'));
    assert.ok(graph.edges.some(e => e.relation === 'HAS_AGE'));
    assert.ok(graph.edges.some(e => e.relation === 'HAS_SEX'));
    assert.ok(graph.edges.some(e => e.relation.startsWith('MENTIONS_')));
    assert.equal(graph.data_source, 'data/interim/cases_clean.csv');
    const visibleMeasurements = graph.nodes.filter(n => n.kind === 'Measurement');
    assert.ok(visibleMeasurements.length > 0);
    assert.ok(visibleMeasurements.every(n => graph.edges.some(e => e.target === n.id && e.source === concept.id)));
    assert.deepEqual(await canvas.evaluate(el => el._cyreg.cy.nodes().map(n => n.id()).sort()), [...ids].sort(),
      'Visible export must match the actual canvas');

    // Native selector provides the same interaction without a pointer.
    const measurement = await page.locator('.kg-picker option').evaluateAll(options =>
      options.find(option => option.textContent.startsWith('Measurement'))?.value);
    assert.ok(measurement);
    await page.locator('.kg-picker').selectOption(measurement);
    await page.waitForFunction(() => document.body.textContent.includes('Selected evidence'));
    assert.equal(await canvas.evaluate(el => el._cyreg.cy.nodes('[kind = "Measurement"]').length), visibleMeasurements.length);
    await page.getByText('Group by IS_A category', { exact: true }).click();
    await page.waitForFunction(() => document.querySelector('.kg-canvas')?._cyreg?.cy?.nodes('[kind = "SemanticType"]').length > 0);
    await page.getByText('Group by IS_A category', { exact: true }).click();
    await page.waitForFunction(() => document.querySelector('.kg-canvas')?._cyreg?.cy?.nodes('[kind = "SemanticType"]').length === 0);
    assert.equal(await canvas.evaluate(el => el._cyreg.cy.nodes('[kind = "Measurement"]').length), 0,
      'Returning to an earlier view must not restore its expanded measurements');
    await page.getByRole('searchbox', { name: 'Find a graph node' }).fill(concept.label);
    await page.getByRole('searchbox', { name: 'Find a graph node' }).press('Enter');
    await page.waitForFunction(() => document.querySelector('.kg-canvas')?._cyreg?.cy?.nodes('[kind = "Measurement"]').length > 0);
    await page.getByRole('button', { name: 'Zoom in', exact: true }).click();
    await page.getByRole('button', { name: 'Zoom out', exact: true }).click();
    await page.getByRole('button', { name: 'Focus selection', exact: true }).click();
    await page.getByRole('button', { name: 'Clear selection', exact: true }).click();
    await page.waitForFunction(() => document.querySelector('.kg-canvas')?._cyreg?.cy?.nodes('[kind = "Measurement"]').length === 0);
    const otherConcept = await canvas.evaluate((el, current) => el._cyreg.cy.nodes()
      .filter(n => n.data('kind') === 'Concept' && n.id() !== current)[0].id(), concept.id);
    await page.locator('.kg-picker').selectOption(otherConcept);
    assert.ok(await canvas.evaluate((el, owner) => el._cyreg.cy.nodes('[kind = "Measurement"]')
      .every(n => n.incomers('node').every(source => source.id() === owner)), otherConcept),
      'Switching concepts must not retain the previous owner measurements');
    await page.locator('.kg-picker').selectOption('');
    await page.waitForFunction(() => document.querySelector('.kg-canvas')?._cyreg?.cy?.nodes('[kind = "Measurement"]').length === 0);
    await page.getByText('Group by IS_A category', { exact: true }).click();
    await page.waitForFunction(() => document.querySelector('.kg-canvas')?._cyreg?.cy?.nodes('[kind = "SemanticType"]').length > 0);
    const grouped = await canvas.evaluate(el => {
      const cy = el._cyreg.cy;
      return { classes: cy.nodes('[kind = "SemanticType"]').map(n => n.data('label')),
        relations: cy.edges('[relation = "IS_A"]').map(e => ({ type: e.data('entity_type'), target: e.target().data('label') })),
        measurements: cy.nodes('[kind = "Measurement"]').length };
    });
    assert.ok(grouped.classes.includes('Treatment'));
    assert.ok(grouped.classes.includes('Diagnosis'));
    assert.ok(grouped.relations.length > 0);
    assert.ok(grouped.relations.every(e => e.type === e.target));
    assert.equal(grouped.measurements, 0);
    await page.getByRole('button', { name: 'Rearrange', exact: true }).click();
    await page.locator('.kg-root').screenshot({ path: process.env.KG_SCREENSHOT || '/tmp/knowledge-graph-smoke.png' });

    await page.getByText('All occurrences', { exact: true }).click();
    await page.locator('.kg-picker option').filter({ hasText: 'Mention ·' }).first().waitFor({ state: 'attached' });
    const stats = await canvas.evaluate(el => {
      const cy = el._cyreg.cy;
      return { nodes: cy.nodes().length, mentions: cy.nodes().filter(n => n.data('kind') === 'Mention').length };
    });
    assert.ok(stats.mentions > 0);
    assert.ok(stats.nodes <= 100);
    assert.equal(errors.length, 0, errors.join('\n'));
    console.log(JSON.stringify({ result: 'passed', visibleNodes: graph.nodes.length,
      selectedConcept: concept.id, occurrenceNodes: stats.nodes, browserErrors: errors }, null, 2));
  } catch (error) {
    await page.screenshot({ path: '/tmp/knowledge-graph-smoke-failure.png', fullPage: true });
    console.error('Browser errors:', errors);
    console.error((await page.locator('body').innerText()).slice(-5000));
    throw error;
  } finally {
    await browser.close();
  }
})();
