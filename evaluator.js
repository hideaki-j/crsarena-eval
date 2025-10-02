// CRS Arena Evaluation Logic
// Ported from eval.py to JavaScript

const TURN_ASPECTS = ["relevance", "interestingness"];
const dialog_ASPECTS = ["understanding", "task_completion", "interest_arousal", "efficiency", "dialog_overall"];
const ALL_ASPECTS = [...TURN_ASPECTS, ...dialog_ASPECTS];
const CHART_LABELS = ALL_ASPECTS.map(aspect => aspect.replace(/_/g, ' '));
const DATASET_ORDER = ["redial", "opendialkg"];
const DATASET_LABELS = {
    redial: "CRSArena-Eval (ReDial)",
    opendialkg: "CRSArena-Eval (OpenDialKG)"
};
const DATASET_CONFIG = {
    redial: {
        baselineKey: "CRSArena-Eval_RD"
    },
    opendialkg: {
        baselineKey: "CRSArena-Eval_KG"
    }
};
const METRIC_LABELS = {
    pearson: "Pearson",
    spearman: "Spearman"
};
const SERIES_STYLES = {
    uploaded: {
        borderColor: 'rgba(102, 126, 234, 1)',
        backgroundColor: 'rgba(102, 126, 234, 0.2)',
        pointBackgroundColor: 'rgba(102, 126, 234, 1)',
        pointHoverBackgroundColor: 'rgba(102, 126, 234, 1)',
        pointHoverBorderColor: 'rgba(102, 126, 234, 1)'
    },
    baseline: {
        borderColor: 'rgba(255, 99, 132, 1)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        pointBackgroundColor: 'rgba(255, 99, 132, 1)',
        pointHoverBackgroundColor: 'rgba(255, 99, 132, 1)',
        pointHoverBorderColor: 'rgba(255, 99, 132, 1)'
    }
};

let goldData = null;
let baselineData = null;
const chartDataCache = {
    redial: { uploaded: null, baselines: {} },
    opendialkg: { uploaded: null, baselines: {} }
};
let comparisonChartInstance = null;
const chartState = {
    metric: 'spearman',
    dataset: 'redial',
    selections: {
        redial: { seriesA: 'uploaded', seriesB: null },
        opendialkg: { seriesA: 'uploaded', seriesB: null }
    }
};

// Statistical functions
function mean(arr) {
    return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function standardDeviation(arr) {
    const avg = mean(arr);
    const squareDiffs = arr.map(value => Math.pow(value - avg, 2));
    return Math.sqrt(mean(squareDiffs));
}

function pearsonCorrelation(x, y) {
    if (x.length !== y.length || x.length === 0) {
        return NaN;
    }
    
    const n = x.length;
    const meanX = mean(x);
    const meanY = mean(y);
    
    let numerator = 0;
    let sumXSquared = 0;
    let sumYSquared = 0;
    
    for (let i = 0; i < n; i++) {
        const diffX = x[i] - meanX;
        const diffY = y[i] - meanY;
        numerator += diffX * diffY;
        sumXSquared += diffX * diffX;
        sumYSquared += diffY * diffY;
    }
    
    const denominator = Math.sqrt(sumXSquared * sumYSquared);
    
    if (denominator === 0) {
        return NaN;
    }
    
    return numerator / denominator;
}

function spearmanCorrelation(x, y) {
    if (x.length !== y.length || x.length === 0) {
        return NaN;
    }
    
    // Create ranks for x and y
    const rankX = getRanks(x);
    const rankY = getRanks(y);
    
    // Calculate Pearson correlation on ranks
    return pearsonCorrelation(rankX, rankY);
}

function getRanks(arr) {
    // Create array of [value, originalIndex]
    const indexed = arr.map((val, idx) => ({ val, idx }));
    
    // Sort by value
    indexed.sort((a, b) => a.val - b.val);
    
    // Assign ranks (handle ties by averaging)
    const ranks = new Array(arr.length);
    let i = 0;
    
    while (i < indexed.length) {
        let j = i;
        // Find all tied values
        while (j < indexed.length && indexed[j].val === indexed[i].val) {
            j++;
        }
        
        // Average rank for tied values
        const rank = (i + j + 1) / 2;
        
        for (let k = i; k < j; k++) {
            ranks[indexed[k].idx] = rank;
        }
        
        i = j;
    }
    
    return ranks;
}

// Data loading functions
async function loadGoldData() {
    try {
        const response = await fetch('crs_arena_eval.json');
        if (!response.ok) {
            throw new Error(`Failed to load gold data: ${response.statusText}`);
        }
        const data = await response.json();
        return parseGoldData(data);
    } catch (error) {
        throw new Error(`Error loading gold data: ${error.message}`);
    }
}

async function loadBaselineData() {
    try {
        const response = await fetch('baselines.json');
        if (!response.ok) {
            throw new Error(`Failed to load baseline data: ${response.statusText}`);
        }
        return await response.json();
    } catch (error) {
        throw new Error(`Error loading baseline data: ${error.message}`);
    }
}

function parseGoldData(goldDataJson) {
    const turnGold = new Map();
    const dialGold = new Map();
    
    for (const dialog of goldDataJson) {
        const convId = dialog.conv_id;
        dialGold.set(convId, dialog.dial_level_aggregated || {});
        
        for (const turn of dialog.dialog || []) {
            if (turn.role !== "ASST") {
                continue;
            }
            const key = `${convId}:${turn.turn_ind}`;
            turnGold.set(key, turn.turn_level_aggregated || {});
        }
    }
    
    return { turnGold, dialGold };
}

function parseRunData(runDataJson) {
    const turnPreds = new Map();
    const dialPreds = new Map();
    
    for (const dialog of runDataJson) {
        const convId = dialog.conv_id;
        const turnList = dialog.turns || [];
        
        for (const turn of turnList) {
            const turnInd = parseInt(turn.turn_ind);
            const key = `${convId}:${turnInd}`;
            const turnLevelPred = {};
            
            for (const aspect of TURN_ASPECTS) {
                if (turn.turn_level_pred && aspect in turn.turn_level_pred) {
                    turnLevelPred[aspect] = parseFloat(turn.turn_level_pred[aspect]);
                }
            }
            
            turnPreds.set(key, turnLevelPred);
        }
        
        const dialLevelPred = {};
        for (const aspect of dialog_ASPECTS) {
            if (dialog.dial_level_pred && aspect in dialog.dial_level_pred) {
                dialLevelPred[aspect] = parseFloat(dialog.dial_level_pred[aspect]);
            }
        }
        dialPreds.set(convId, dialLevelPred);
    }
    
    return { turnPreds, dialPreds };
}

function datasetFromConvId(convId) {
    const parts = convId.split('_');
    if (parts.length < 2) {
        throw new Error(`Unexpected conv_id format: ${convId}`);
    }
    return parts[1];
}

function systemFromConvId(convId) {
    const parts = convId.split('_');
    if (parts.length < 2) {
        throw new Error(`Unexpected conv_id format: ${convId}`);
    }
    return `${parts[0]}_${parts[1]}`;
}

function computePerSystemSpearman(turnPreds, turnGold, dialPreds, dialGold) {
    const aspects = [...TURN_ASPECTS, ...dialog_ASPECTS];
    const bySystem = new Map();

    function ensureSystem(systemId) {
        if (!bySystem.has(systemId)) {
            const aspectBuckets = {};
            for (const aspect of aspects) {
                aspectBuckets[aspect] = { pred: [], gold: [] };
            }
            bySystem.set(systemId, aspectBuckets);
        }
    }

    for (const [key, goldAspects] of turnGold) {
        const [convId] = key.split(':');
        if (!turnPreds.has(key)) {
            continue;
        }
        const predAspects = turnPreds.get(key);
        const systemId = systemFromConvId(convId);
        ensureSystem(systemId);

        for (const aspect of TURN_ASPECTS) {
            if (!(aspect in goldAspects) || !(aspect in predAspects)) {
                continue;
            }
            const bucket = bySystem.get(systemId)[aspect];
            bucket.pred.push(predAspects[aspect]);
            bucket.gold.push(goldAspects[aspect]);
        }
    }

    for (const [convId, goldAspects] of dialGold) {
        if (!dialPreds.has(convId)) {
            continue;
        }
        const predAspects = dialPreds.get(convId);
        const systemId = systemFromConvId(convId);
        ensureSystem(systemId);

        for (const aspect of dialog_ASPECTS) {
            if (!(aspect in goldAspects) || !(aspect in predAspects)) {
                continue;
            }
            const bucket = bySystem.get(systemId)[aspect];
            bucket.pred.push(predAspects[aspect]);
            bucket.gold.push(goldAspects[aspect]);
        }
    }

    const systemLabels = Array.from(bySystem.keys()).sort();
    const values = systemLabels.map(systemId => {
        const aspectBuckets = bySystem.get(systemId);
        return aspects.map(aspect => {
            const bucket = aspectBuckets[aspect];
            if (!bucket || bucket.pred.length === 0) {
                return NaN;
            }
            return spearmanCorrelation(bucket.pred, bucket.gold);
        });
    });

    return {
        labels: systemLabels,
        aspects,
        values
    };
}

function computeMetrics(records) {
    const byDataset = {};
    
    for (const dataset of DATASET_ORDER) {
        byDataset[dataset] = { pred: [], gold: [] };
    }
    const aggregate = { pred: [], gold: [] };
    
    for (const { dataset, pred, gold } of records) {
        if (!byDataset[dataset]) {
            continue;
        }
        byDataset[dataset].pred.push(pred);
        byDataset[dataset].gold.push(gold);
        aggregate.pred.push(pred);
        aggregate.gold.push(gold);
    }
    
    const metrics = {};
    
    for (const dataset of DATASET_ORDER) {
        const preds = byDataset[dataset].pred;
        const golds = byDataset[dataset].gold;
        
        if (preds.length === 0) {
            metrics[dataset] = { pearson: NaN, spearman: NaN };
            continue;
        }
        
        const pearson = pearsonCorrelation(preds, golds);
        const spearman = spearmanCorrelation(preds, golds);
        
        metrics[dataset] = { pearson, spearman };
    }

    if (aggregate.pred.length === 0) {
        metrics.all = { pearson: NaN, spearman: NaN };
    } else {
        metrics.all = {
            pearson: pearsonCorrelation(aggregate.pred, aggregate.gold),
            spearman: spearmanCorrelation(aggregate.pred, aggregate.gold)
        };
    }
    
    return metrics;
}

function buildSeriesFromResults(turnResults, dialogResults, datasetKey) {
    const series = { pearson: [], spearman: [] };

    for (const aspect of TURN_ASPECTS) {
        const stats = turnResults[aspect] || {};
        const datasetStats = stats[datasetKey] || { pearson: NaN, spearman: NaN };
        series.pearson.push(formatNumber(datasetStats.pearson));
        series.spearman.push(formatNumber(datasetStats.spearman));
    }

    for (const aspect of dialog_ASPECTS) {
        const stats = dialogResults[aspect] || {};
        const datasetStats = stats[datasetKey] || { pearson: NaN, spearman: NaN };
        series.pearson.push(formatNumber(datasetStats.pearson));
        series.spearman.push(formatNumber(datasetStats.spearman));
    }

    return series;
}

function buildBaselineSeries(baselineMetrics) {
    const series = { pearson: [], spearman: [] };

    for (const aspect of ALL_ASPECTS) {
        const metrics = baselineMetrics && baselineMetrics[aspect] ? baselineMetrics[aspect] : {};
        series.pearson.push(formatNumber(metrics.pearson_r));
        series.spearman.push(formatNumber(metrics.spearman_rho));
    }

    return series;
}

function processBaselineData(rawBaselines) {
    if (!rawBaselines) {
        return;
    }

    for (const [datasetKey, config] of Object.entries(DATASET_CONFIG)) {
        const baselineKey = config.baselineKey;
        const datasetBaselines = rawBaselines[baselineKey] || {};
        const seriesByBaseline = {};

        for (const [baselineName, metrics] of Object.entries(datasetBaselines)) {
            seriesByBaseline[baselineName] = buildBaselineSeries(metrics);
        }

        chartDataCache[datasetKey].baselines = seriesByBaseline;
        ensureSelectionForDataset(datasetKey);
    }
}

function evaluateTurnLevel(turnPreds, turnGold) {
    const results = {};
    
    for (const aspect of TURN_ASPECTS) {
        const records = [];
        
        for (const [key, goldAspects] of turnGold) {
            if (!(aspect in goldAspects)) {
                continue;
            }
            
            if (!turnPreds.has(key)) {
                continue;
            }
            
            const predAspects = turnPreds.get(key);
            if (!(aspect in predAspects)) {
                continue;
            }
            
            const convId = key.split(':')[0];
            const dataset = datasetFromConvId(convId);
            
            records.push({
                dataset,
                pred: predAspects[aspect],
                gold: goldAspects[aspect]
            });
        }
        
        results[aspect] = computeMetrics(records);
    }
    
    return results;
}

function evaluatedialogLevel(dialPreds, dialGold) {
    const results = {};
    
    for (const aspect of dialog_ASPECTS) {
        const records = [];
        
        for (const [convId, goldAspects] of dialGold) {
            if (!(aspect in goldAspects)) {
                continue;
            }
            
            if (!dialPreds.has(convId)) {
                continue;
            }
            
            const predAspects = dialPreds.get(convId);
            if (!(aspect in predAspects)) {
                continue;
            }
            
            const dataset = datasetFromConvId(convId);
            
            records.push({
                dataset,
                pred: predAspects[aspect],
                gold: goldAspects[aspect]
            });
        }
        
        results[aspect] = computeMetrics(records);
    }
    
    return results;
}

function formatNumber(num) {
    if (num === null || num === undefined) {
        return 0;
    }
    const value = Number(num);
    if (Number.isNaN(value)) {
        return 0; // Use 0 for N/A values in charts
    }
    return parseFloat(value.toFixed(3));
}

function formatTableValue(value) {
    if (value === null || value === undefined) {
        return '—';
    }
    const numeric = Number(value);
    if (Number.isNaN(numeric)) {
        return '—';
    }
    return numeric.toFixed(3);
}

function displayResults(turnResults, dialogResults) {
    // Show results section first so canvas elements are visible
    document.getElementById('results').style.display = 'block';

    const metricSelect = document.getElementById('chartMetricSelect');
    if (!metricSelect) {
        console.error('Metric select element not found!');
        showError('Chart elements not found. Please refresh the page.');
        return;
    }

    chartDataCache.redial.uploaded = buildSeriesFromResults(turnResults, dialogResults, 'redial');
    chartDataCache.opendialkg.uploaded = buildSeriesFromResults(turnResults, dialogResults, 'opendialkg');

    chartState.selections.redial.seriesA = 'uploaded';
    chartState.selections.opendialkg.seriesA = 'uploaded';

    ensureSelectionForDataset('redial');
    ensureSelectionForDataset('opendialkg');

    populateSourceSelects();
    renderComparisonChart();
}

function ensureSelectionForDataset(datasetKey) {
    const cache = chartDataCache[datasetKey] || {};
    let selection = chartState.selections[datasetKey];
    if (!selection) {
        selection = { seriesA: null, seriesB: null };
    }

    selection.seriesA = cache.uploaded ? 'uploaded' : null;

    const baselineKeys = Object.keys(cache.baselines || {}).sort();
    if (baselineKeys.length === 0) {
        selection.seriesB = null;
    } else if (!baselineKeys.includes(selection.seriesB)) {
        selection.seriesB = baselineKeys[0];
    }

    chartState.selections[datasetKey] = selection;
    return { selection, baselineKeys };
}

function populateSourceSelects() {
    const datasetKey = chartState.dataset;
    const datasetSelect = document.getElementById('datasetSelect');
    if (datasetSelect) {
        datasetSelect.value = datasetKey;
    }

    const chartTitle = document.querySelector('.chart-header h3');
    if (chartTitle) {
        const datasetLabel = DATASET_LABELS[datasetKey] || datasetKey;
        chartTitle.textContent = `${datasetLabel} Comparison`;
    }

    const selectB = document.getElementById('sourceBSelect');
    if (!selectB) {
        return;
    }

    const { selection, baselineKeys } = ensureSelectionForDataset(datasetKey);

    selectB.innerHTML = '';

    if (baselineKeys.length === 0) {
        selectB.disabled = true;
        chartState.selections[datasetKey].seriesB = null;
        selectB.value = '';
    } else {
        selectB.disabled = false;
        for (const key of baselineKeys) {
            const option = document.createElement('option');
            option.value = key;
            option.textContent = formatSourceLabel(key);
            selectB.appendChild(option);
        }
        selectB.value = selection.seriesB;
    }
}

function renderComparisonChart() {
    const canvas = document.getElementById('comparisonChart');
    if (!canvas) {
        console.error('Comparison chart canvas not found!');
        return;
    }

    const datasetKey = chartState.dataset;
    const { selection, baselineKeys } = ensureSelectionForDataset(datasetKey);
    const metric = chartState.metric;

    const datasets = [];

    const uploadedSeries = resolveSeries(datasetKey, 'uploaded');
    if (uploadedSeries) {
        const uploadData = uploadedSeries[metric];
        if (uploadData && uploadData.length > 0) {
            const style = SERIES_STYLES.uploaded;
            datasets.push({
                label: 'Your evaluator',
                data: uploadData,
                borderColor: style.borderColor,
                backgroundColor: style.backgroundColor,
                borderWidth: 2,
                pointBackgroundColor: style.pointBackgroundColor,
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: style.pointHoverBackgroundColor,
                pointHoverBorderColor: style.pointHoverBorderColor
            });
        }
    }

    const baselineKey = selection.seriesB;
    let baselineSeries = null;
    if (baselineKey) {
        baselineSeries = resolveSeries(datasetKey, baselineKey);
        if (baselineSeries) {
            const baselineData = baselineSeries[metric];
            if (baselineData && baselineData.length > 0) {
                const style = SERIES_STYLES.baseline;
                datasets.push({
                    label: `Baseline - ${formatSourceLabel(baselineKey)}`,
                    data: baselineData,
                    borderColor: style.borderColor,
                    backgroundColor: style.backgroundColor,
                    borderWidth: 2,
                    pointBackgroundColor: style.pointBackgroundColor,
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: style.pointHoverBackgroundColor,
                    pointHoverBorderColor: style.pointHoverBorderColor
                });
            }
        }
    }

    renderComparisonTable(datasetKey, metric, uploadedSeries, baselineSeries, baselineKey, baselineKeys ? baselineKeys.length : 0);

    if (datasets.length === 0) {
        if (comparisonChartInstance) {
            comparisonChartInstance.destroy();
            comparisonChartInstance = null;
        }
        return;
    }

    if (comparisonChartInstance) {
        comparisonChartInstance.destroy();
    }

    const ctx = canvas.getContext('2d');
    comparisonChartInstance = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: CHART_LABELS,
            datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1,
                    min: 0,
                    ticks: {
                        stepSize: 0.2,
                        callback: function(value) {
                            return value.toFixed(1);
                        }
                    },
                    pointLabels: {
                        font: {
                            size: 12
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + context.parsed.r.toFixed(3);
                        }
                    }
                }
            }
        }
    });
}

function resolveSeries(datasetKey, sourceKey) {
    if (!sourceKey) {
        return null;
    }
    const cache = chartDataCache[datasetKey];
    if (!cache) {
        return null;
    }
    if (sourceKey === 'uploaded') {
        return cache.uploaded;
    }
    return cache.baselines[sourceKey];
}

function formatSourceLabel(sourceKey) {
    if (sourceKey === 'uploaded') {
        return 'Your evaluator';
    }
    return sourceKey || 'N/A';
}

function renderComparisonTable(datasetKey, metric, uploadedSeries, baselineSeries, baselineKey, baselineCount) {
    const tableBody = document.getElementById('comparisonTableBody');
    const baselineHeader = document.getElementById('comparisonBaselineHeader');
    const note = document.getElementById('comparisonTableNote');

    if (!tableBody || !baselineHeader || !note) {
        return;
    }

    baselineHeader.textContent = baselineKey ? `Baseline - ${formatSourceLabel(baselineKey)}` : 'Baseline';
    note.textContent = '';
    tableBody.innerHTML = '';

    const yourData = uploadedSeries ? uploadedSeries[metric] : null;
    const baselineData = baselineSeries ? baselineSeries[metric] : null;

    if (!yourData || yourData.length === 0) {
        note.textContent = 'No data available for your evaluator.';
        return;
    }

    CHART_LABELS.forEach((label, index) => {
        const row = document.createElement('tr');

        const aspectCell = document.createElement('td');
        aspectCell.textContent = label;

        const yourCell = document.createElement('td');
        yourCell.textContent = formatTableValue(yourData[index]);

        const baselineCell = document.createElement('td');
        baselineCell.textContent = baselineData ? formatTableValue(baselineData[index]) : '—';

        row.appendChild(aspectCell);
        row.appendChild(yourCell);
        row.appendChild(baselineCell);

        tableBody.appendChild(row);
    });

    if (!baselineKey) {
        note.textContent = baselineCount === 0
            ? 'No baselines are available for this dataset.'
            : 'Select a baseline to populate the comparison column.';
    } else if (!baselineData) {
        note.textContent = 'Baseline data unavailable for the selected evaluator.';
    }
}

function showError(message) {
    const errorDiv = document.getElementById('error');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    
    setTimeout(() => {
        errorDiv.style.display = 'none';
    }, 5000);
}

function showLoading(show) {
    document.getElementById('loading').style.display = show ? 'block' : 'none';
}

function showFileInfo(filename, size) {
    const fileInfoDiv = document.getElementById('fileInfo');
    fileInfoDiv.innerHTML = `
        <strong>File loaded:</strong> ${filename} (${(size / 1024).toFixed(2)} KB)
    `;
    fileInfoDiv.style.display = 'block';
}

async function processFile(file) {
    try {
        showLoading(true);
        document.getElementById('error').style.display = 'none';
        document.getElementById('results').style.display = 'none';
        
        // Load gold data if not already loaded
        if (!goldData) {
            goldData = await loadGoldData();
        }
        
        // Read and parse the uploaded file
        const fileContent = await file.text();
        const runDataJson = JSON.parse(fileContent);
        
        showFileInfo(file.name, file.size);
        
        // Parse run data
        const { turnPreds, dialPreds } = parseRunData(runDataJson);
        
        // Evaluate
        const turnResults = evaluateTurnLevel(turnPreds, goldData.turnGold);
        const dialogResults = evaluatedialogLevel(dialPreds, goldData.dialGold);
        
        // Display results
        displayResults(turnResults, dialogResults);
        showLoading(false);
        
    } catch (error) {
        console.error('Error processing file:', error);
        showError(`Error: ${error.message}`);
        showLoading(false);
    }
}

// Event handlers
document.getElementById('fileInput').addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        processFile(file);
    }
});

// Drag and drop functionality
const uploadSection = document.getElementById('uploadSection');

uploadSection.addEventListener('dragover', (event) => {
    event.preventDefault();
    uploadSection.classList.add('dragover');
});

uploadSection.addEventListener('dragleave', () => {
    uploadSection.classList.remove('dragover');
});

uploadSection.addEventListener('drop', (event) => {
    event.preventDefault();
    uploadSection.classList.remove('dragover');
    
    const file = event.dataTransfer.files[0];
    if (file && file.type === 'application/json') {
        processFile(file);
    } else {
        showError('Please upload a valid JSON file.');
    }
});

// Initialize: Load data and wire controls on page load
window.addEventListener('DOMContentLoaded', async () => {
    const datasetSelect = document.getElementById('datasetSelect');
    const metricSelect = document.getElementById('chartMetricSelect');
    const sourceBSelect = document.getElementById('sourceBSelect');

    if (datasetSelect) {
        datasetSelect.value = chartState.dataset;
        datasetSelect.addEventListener('change', (event) => {
            chartState.dataset = event.target.value;
            populateSourceSelects();
            renderComparisonChart();
        });
    }

    if (metricSelect) {
        metricSelect.value = chartState.metric;
        metricSelect.addEventListener('change', (event) => {
            chartState.metric = event.target.value;
            renderComparisonChart();
        });
    }

    if (sourceBSelect) {
        sourceBSelect.addEventListener('change', (event) => {
            const datasetKey = chartState.dataset;
            if (!chartState.selections[datasetKey]) {
                chartState.selections[datasetKey] = { seriesA: null, seriesB: null };
            }
            chartState.selections[datasetKey].seriesB = event.target.value || null;
            renderComparisonChart();
        });
    }

    try {
        goldData = await loadGoldData();
        console.log('Gold data loaded successfully');
    } catch (error) {
        console.error('Failed to load gold data:', error);
        showError('Failed to load evaluation data. Please check if crs_arena_eval.json is available.');
    }

    try {
        baselineData = await loadBaselineData();
        processBaselineData(baselineData);
        console.log('Baseline data loaded successfully');
    } catch (error) {
        console.error('Failed to load baseline data:', error);
        showError('Failed to load baseline evaluator data. Baseline comparisons may be unavailable.');
    }

    populateSourceSelects();
    renderComparisonChart();
});
